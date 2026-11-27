import pytest
from fastapi.testclient import TestClient
from main import app
from database import SessionLocal, User, TransactionRecord

client = TestClient(app)


def test_health_and_root():
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert "Retail Intelligence" in data["message"]
    assert data["version"] == "2.0.0"

    h = client.get("/health")
    assert h.status_code == 200
    assert h.json()["status"] == "healthy"


def test_auth_registration_and_login():
    # 1. Register new analyst
    reg_payload = {
        "name": "Test Analyst",
        "email": "test_analyst@retail.com",
        "password": "Password123",
        "role": "Analyst",
    }
    reg_res = client.post("/auth/register", json=reg_payload)
    assert reg_res.status_code in (201, 400)  # 201 if first time, 400 if already exists

    # 2. Login with registered user
    login_payload = {
        "email": "test_analyst@retail.com",
        "password": "Password123",
    }
    login_res = client.post("/auth/login", json=login_payload)
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]
    assert token is not None

    # 3. Test /auth/me with Bearer token
    headers = {"Authorization": f"Bearer {token}"}
    me_res = client.get("/auth/me", headers=headers)
    assert me_res.status_code == 200
    assert me_res.json()["email"] == "test_analyst@retail.com"
    assert me_res.json()["role"] == "Analyst"


def test_auth_invalid_credentials():
    bad_login = {
        "email": "nonexistent@retail.com",
        "password": "WrongPassword",
    }
    res = client.post("/auth/login", json=bad_login)
    assert res.status_code == 401


def test_role_based_access_control():
    # Login as Admin
    admin_login = client.post("/auth/login", json={"email": "admin@retail.com", "password": "Admin@123"})
    assert admin_login.status_code == 200
    admin_token = admin_login.json()["access_token"]
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    # Admin accesses /auth/users -> 200 OK
    users_res = client.get("/auth/users", headers=admin_headers)
    assert users_res.status_code == 200
    assert len(users_res.json()) >= 3

    # Login as Analyst
    analyst_login = client.post("/auth/login", json={"email": "analyst@retail.com", "password": "Analyst@123"})
    assert analyst_login.status_code == 200
    analyst_token = analyst_login.json()["access_token"]
    analyst_headers = {"Authorization": f"Bearer {analyst_token}"}

    # Analyst tries to access /auth/users -> 403 Forbidden
    forbidden_res = client.get("/auth/users", headers=analyst_headers)
    assert forbidden_res.status_code == 403


def test_crud_full_cycle():
    admin_login = client.post("/auth/login", json={"email": "admin@retail.com", "password": "Admin@123"})
    admin_token = admin_login.json()["access_token"]
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    # 1. CREATE
    create_payload = {
        "invoice_no": "INV-TEST-999",
        "stock_code": "TEST-SKU-999",
        "description": "Integration Test Product",
        "quantity": 5.0,
        "unit_price": 20.0,
        "customer_id": "99999",
        "country": "United Kingdom",
    }
    c_res = client.post("/transactions", json=create_payload, headers=admin_headers)
    assert c_res.status_code == 201
    tx_id = c_res.json()["transaction"]["id"]
    assert c_res.json()["transaction"]["total_amount"] == 100.0

    # 2. READ
    r_res = client.get(f"/transactions/{tx_id}")
    assert r_res.status_code == 200
    assert r_res.json()["stock_code"] == "TEST-SKU-999"

    # 3. UPDATE (PUT / PATCH)
    update_payload = {
        "quantity": 10.0,
        "unit_price": 25.0,
        "description": "Updated Test Product Description",
    }
    u_res = client.put(f"/transactions/{tx_id}", json=update_payload, headers=admin_headers)
    assert u_res.status_code == 200
    assert u_res.json()["transaction"]["total_amount"] == 250.0
    assert u_res.json()["transaction"]["description"] == "Updated Test Product Description"

    # 4. DELETE
    d_res = client.delete(f"/transactions/{tx_id}", headers=admin_headers)
    assert d_res.status_code == 200
    assert d_res.json()["status"] == "deleted"

    # Verify 404 after deletion
    not_found = client.get(f"/transactions/{tx_id}")
    assert not_found.status_code == 404


def test_validation_errors():
    # 422 Unprocessable Entity on missing required field
    bad_payload = {"invoice_no": "INV-X"}  # missing stock_code, quantity, unit_price
    res = client.post("/transactions", json=bad_payload)
    assert res.status_code == 422

    # 400 Bad Request on empty batch
    res_batch = client.post("/transactions/batch", json={"transactions": []})
    assert res_batch.status_code == 400


def test_analytics_and_predictions():
    # Analytics summary
    sum_res = client.get("/analytics/summary")
    assert sum_res.status_code == 200
    assert "total_transactions" in sum_res.json()

    # Product list
    prod_res = client.get("/analytics/products?limit=5")
    assert prod_res.status_code == 200

    # Forecasts
    sales_fc = client.get("/forecast/sales?days=7")
    assert sales_fc.status_code == 200
    assert len(sales_fc.json()) == 7

    canc_fc = client.get("/forecast/cancellations?days=7")
    assert canc_fc.status_code == 200
    assert len(canc_fc.json()) == 7

    # Financial impact simulator
    fi_res = client.post("/financial-impact", json={"total_credit": 5000.0, "reduction_pct": 0.25})
    assert fi_res.status_code == 200
    assert fi_res.json()["potential_saving"] == 1250.0
