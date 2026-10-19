"""Encrypted file storage for uploaded resumes, plus file-type sniffing.

Files on disk are AES-256-GCM encrypted:  b"VIE1" | key id length (1 byte) | key id | 12-byte nonce | ciphertext+tag
The key id lets old files be read after the key is rotated (see FILE_ENCRYPTION_KEYS in config.py).
"""
import base64
import hashlib
import io
import os
import secrets
import uuid
import zipfile
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from flask import current_app

MAGIC = b"VIE1"


class StorageError(Exception):
    pass


class FileTypeError(ValueError):
    pass


# ---------- keys ----------

def parse_keyring(value):
    """'id:base64key,id2:base64key' -> [(id, key bytes)], first = current."""
    keys = []
    for item in (value or "").split(","):
        item = item.strip()
        if not item:
            continue
        key_id, sep, encoded = item.partition(":")
        if not sep or not key_id or len(key_id) > 32:
            raise StorageError(f"Malformed FILE_ENCRYPTION_KEYS entry {key_id!r} (expected id:base64key)")
        try:
            key = base64.b64decode(encoded, validate=True)
        except Exception as e:
            raise StorageError(f"Encryption key {key_id!r} is not valid base64") from e
        if len(key) != 32:
            raise StorageError(f"Encryption key {key_id!r} must be 32 bytes (got {len(key)})")
        keys.append((key_id, key))
    if not keys:
        raise StorageError("FILE_ENCRYPTION_KEYS is not set; resumes can't be stored. "
                           "Run: flask --app run generate-encryption-key")
    return keys


def generate_key_entry(key_id=None):
    key_id = key_id or f"k{secrets.token_hex(3)}"
    return f"{key_id}:{base64.b64encode(os.urandom(32)).decode()}"


def _keyring():
    return parse_keyring(current_app.config["FILE_ENCRYPTION_KEYS"])


# ---------- encryption ----------

def encrypt(data):
    key_id, key = _keyring()[0]
    nonce = os.urandom(12)
    kid = key_id.encode()
    header = MAGIC + bytes([len(kid)]) + kid
    return header + nonce + AESGCM(key).encrypt(nonce, data, header)  # header is authenticated too


def is_encrypted(blob):
    return blob[:4] == MAGIC


def decrypt(blob):
    if not is_encrypted(blob):
        raise StorageError("File is not encrypted with this system's format")
    kid_len = blob[4]
    key_id = blob[5:5 + kid_len].decode(errors="replace")
    header, rest = blob[:5 + kid_len], blob[5 + kid_len:]
    keys = dict(_keyring())
    if key_id not in keys:
        raise StorageError(f"No key {key_id!r} in FILE_ENCRYPTION_KEYS to decrypt this file")
    try:
        return AESGCM(keys[key_id]).decrypt(rest[:12], rest[12:], header)
    except InvalidTag as e:
        raise StorageError("File is corrupted or was tampered with (authentication failed)") from e


# ---------- storage ----------

def _folder():
    folder = Path(current_app.config["UPLOAD_FOLDER"])
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_resume(data, extension):
    """Encrypt and store; returns the stored file name (random, never derived from user input)."""
    name = f"{uuid.uuid4().hex}.{extension}.enc"
    path = _folder() / name
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(encrypt(data))
    tmp.replace(path)  # atomic: a crash never leaves a half-written file under the real name
    return name


def read_resume(stored_name):
    path = _folder() / Path(stored_name).name  # never follow path components from the database
    if not path.exists():
        raise StorageError("The stored resume file is missing")
    blob = path.read_bytes()
    return decrypt(blob) if is_encrypted(blob) else blob  # plaintext = legacy file from before encryption


def delete_resume(stored_name):
    if stored_name:
        (_folder() / Path(stored_name).name).unlink(missing_ok=True)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


# ---------- file type sniffing ----------

def detect_resume_type(data, filename):
    """Returns 'pdf' or 'docx' based on the actual bytes (not just the extension). Raises FileTypeError."""
    ext = Path(filename or "").suffix.lower().lstrip(".")
    if not data:
        raise FileTypeError("The file is empty.")
    if data[:5] == b"%PDF-":
        kind = "pdf"
    elif data[:4] == b"PK\x03\x04":
        kind = _check_docx(data)
    else:
        raise FileTypeError("This doesn't look like a PDF or Word (DOCX) file.")
    if ext != kind:
        raise FileTypeError(f"The file's contents are a {kind.upper()} but its name ends in .{ext or '?'}. "
                            f"Please rename it to .{kind} and try again.")
    return kind


def _check_docx(data):
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = set(zf.namelist())
            total = sum(info.file_size for info in zf.infolist())
            content_types = zf.read("[Content_Types].xml") if "[Content_Types].xml" in names else b""
    except zipfile.BadZipFile as e:
        raise FileTypeError("The Word file is damaged and can't be opened.") from e
    if total > current_app.config["MAX_DOCX_UNCOMPRESSED"]:
        raise FileTypeError("The Word file is too large once unpacked.")
    if "word/document.xml" not in names:
        raise FileTypeError("This ZIP file isn't a Word document.")
    if b"macroEnabled" in content_types or "word/vbaProject.bin" in names:
        raise FileTypeError("Macro-enabled Word files aren't accepted. Please save it as a normal .docx.")
    return "docx"
