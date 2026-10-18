import re

import pytest

from app.extensions import db
from app.models import AuditLog, User
from tests.conftest import PASSWORD


@pytest.fixture
def admin_client(make_user, login):
    make_user("root@test.com", role=User.ROLE_ADMIN, name="Root Admin")
    return login("root@test.com")


def uid(app, email):
    with app.app_context():
        return User.query.filter_by(email=email).one().id


def get(app, email):
    with app.app_context():
        u = User.query.filter_by(email=email).one()
        db.session.expunge(u)
        return u


def test_admin_pages_are_admin_only(app, hr_client, candidate_client, admin_client):
    for client in (hr_client, candidate_client):
        assert client.get("/admin/").status_code == 403
        assert client.get("/admin/audit").status_code == 403
    assert admin_client.get("/admin/").status_code == 200
    assert admin_client.get("/hr/").status_code == 200  # admins also have HR access
    assert admin_client.get("/dashboard").headers["Location"].endswith("/admin/")


def test_user_list_search_and_filters(app, admin_client, make_user):
    make_user("asha@test.com", name="Asha Kumar")
    make_user("ravi@test.com", name="Ravi", verified=False)
    make_user("hr2@test.com", role=User.ROLE_HR, name="Hira")
    page = admin_client.get("/admin/?q=KUMAR").data
    assert b"asha@test.com" in page and b"ravi@test.com" not in page
    assert b"hr2@test.com" in admin_client.get("/admin/?role=hr").data
    assert b"asha@test.com" not in admin_client.get("/admin/?role=hr").data
    assert b"ravi@test.com" in admin_client.get("/admin/?status=unverified").data
    assert b"No users match" in admin_client.get("/admin/?q=%25").data  # % is literal, not a wildcard


def test_invite_staff_and_accept(app, admin_client, outbox):
    response = admin_client.post("/admin/users/invite", data={"name": "Hira HR", "email": "Hira@Test.com", "role": "hr"},
                                 follow_redirects=True)
    assert b"Invitation sent to hira@test.com" in response.data
    invited = get(app, "hira@test.com")
    assert invited.role == User.ROLE_HR and not invited.is_verified and invited.last_login_at is None
    link = re.search(r"http://localhost(/\S+)", outbox[0]["text"]).group(1)
    assert "invited you" in outbox[0]["text"] and "Root Admin" in outbox[0]["text"]

    client = app.test_client()
    assert b"Welcome, Hira HR" in client.get(link).data
    response = client.post(link, data={"password": "Recruit-Strong-9", "confirm": "Recruit-Strong-9"}, follow_redirects=True)
    assert b"Your password has been set" in response.data
    assert get(app, "hira@test.com").is_verified
    assert client.post("/login", data={"email": "hira@test.com", "password": "Recruit-Strong-9"}).status_code == 302
    assert b"already been used" in app.test_client().get(link, follow_redirects=True).data


def test_invite_validation(app, admin_client, make_user):
    make_user("taken@test.com")
    assert b"already exists" in admin_client.post("/admin/users/invite", data={"name": "X", "email": "taken@test.com", "role": "hr"}).data
    assert b"Not a valid choice" in admin_client.post("/admin/users/invite", data={"name": "X", "email": "x@test.com", "role": "candidate"}).data


def test_deactivate_and_reactivate(app, admin_client, make_user, login):
    make_user("asha@test.com")
    asha = login("asha@test.com")
    user_id = uid(app, "asha@test.com")
    admin_client.post(f"/admin/users/{user_id}/deactivate")
    assert not get(app, "asha@test.com").active
    assert asha.get("/candidate/").status_code == 302  # signed out immediately
    admin_client.post(f"/admin/users/{user_id}/activate")
    assert get(app, "asha@test.com").active
    assert login("asha@test.com")


def test_admin_cannot_lock_themselves_out(app, admin_client):
    me = uid(app, "root@test.com")
    assert b"can&#39;t deactivate your own" in admin_client.post(f"/admin/users/{me}/deactivate", follow_redirects=True).data
    assert b"can&#39;t change your own role" in admin_client.post(f"/admin/users/{me}/role", data={"role": "hr"}, follow_redirects=True).data
    assert b"from Account settings" in admin_client.post(f"/admin/users/{me}/delete", data={"confirm_email": "root@test.com"},
                                                        follow_redirects=True).data
    assert get(app, "root@test.com").active and get(app, "root@test.com").is_admin


def test_last_admin_protection_with_two_admins(app, admin_client, make_user, login):
    make_user("second@test.com", role=User.ROLE_ADMIN)
    second = login("second@test.com")
    root_id = uid(app, "root@test.com")
    # second admin demotes root: allowed (two admins exist)
    second.post(f"/admin/users/{root_id}/role", data={"role": "hr"})
    assert get(app, "root@test.com").role == User.ROLE_HR
    # now second is the only admin; root (HR) can't reach admin pages anymore
    assert admin_client.get("/admin/").status_code in (302, 403)


def test_role_change_rules(app, admin_client, make_user):
    make_user("cand@x.com")
    make_user("hr@x.com", role=User.ROLE_HR)
    response = admin_client.post(f"/admin/users/{uid(app, 'cand@x.com')}/role", data={"role": "hr"}, follow_redirects=True)
    assert b"can&#39;t be turned into staff" in response.data
    assert admin_client.post(f"/admin/users/{uid(app, 'hr@x.com')}/role", data={"role": "superuser"}).status_code == 400
    admin_client.post(f"/admin/users/{uid(app, 'hr@x.com')}/role", data={"role": "admin"})
    assert get(app, "hr@x.com").is_admin


def test_unlock_and_force_sign_out(app, admin_client, make_user, login):
    make_user("asha@test.com")
    asha = login("asha@test.com")
    for _ in range(5):
        app.test_client().post("/login", data={"email": "asha@test.com", "password": "nope-nope-1"})
    assert get(app, "asha@test.com").is_locked
    admin_client.post(f"/admin/users/{uid(app, 'asha@test.com')}/unlock")
    assert not get(app, "asha@test.com").is_locked
    admin_client.post(f"/admin/users/{uid(app, 'asha@test.com')}/sign-out")
    assert asha.get("/candidate/").status_code == 302


def test_resend_invite_only_for_pending_staff(app, admin_client, make_user, outbox):
    make_user("cand@x.com")
    admin_client.post("/admin/users/invite", data={"name": "Hira", "email": "hira@test.com", "role": "hr"})
    outbox.clear()
    admin_client.post(f"/admin/users/{uid(app, 'hira@test.com')}/resend-invite")
    assert len(outbox) == 1
    response = admin_client.post(f"/admin/users/{uid(app, 'cand@x.com')}/resend-invite", follow_redirects=True)
    assert b"Only staff who haven" in response.data and len(outbox) == 1


def test_admin_deletes_user_with_confirmation(app, admin_client, make_user):
    make_user("asha@test.com")
    user_id = uid(app, "asha@test.com")
    response = admin_client.post(f"/admin/users/{user_id}/delete", data={"confirm_email": "wrong@test.com"}, follow_redirects=True)
    assert b"Type the user" in response.data
    response = admin_client.post(f"/admin/users/{user_id}/delete", data={"confirm_email": "ASHA@test.com"}, follow_redirects=True)
    assert b"permanently deleted" in response.data
    with app.app_context():
        assert User.query.filter_by(email="asha@test.com").first() is None
        entry = AuditLog.query.filter_by(action="account.deleted").one()
        assert entry.actor_email == "root@test.com" and entry.details["reason"] == "deleted by admin"


def test_user_detail_and_audit_log(app, admin_client, make_user):
    make_user("asha@test.com")
    app.test_client().post("/login", data={"email": "asha@test.com", "password": "nope-nope-1"})
    page = admin_client.get(f"/admin/users/{uid(app, 'asha@test.com')}").data
    assert b"login.failed" in page
    log = admin_client.get("/admin/audit?action=login.failed").data
    assert b"asha@test.com" in log
    assert b"asha@test.com" not in admin_client.get("/admin/audit?action=account.deleted").data
    assert b"asha@test.com" in admin_client.get("/admin/audit?email=ASHA").data


def test_admin_actions_on_missing_user_return_404(admin_client):
    assert admin_client.get("/admin/users/9999").status_code == 404
    assert admin_client.post("/admin/users/9999/deactivate").status_code == 404


def test_audit_pagination(app, admin_client):
    with app.app_context():
        for i in range(60):
            db.session.add(AuditLog(action="test.event", target_email=f"u{i}@test.com", details={}))
        db.session.commit()
    first = admin_client.get("/admin/audit?action=test.event").data
    second = admin_client.get("/admin/audit?action=test.event&page=2").data
    assert b"u59@test.com" in first and b"u0@test.com" not in first
    assert b"u0@test.com" in second
    assert admin_client.get("/admin/audit?page=999").status_code == 200
