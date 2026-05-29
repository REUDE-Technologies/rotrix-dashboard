"""
Seed script: create the initial super_admin user and organization.

Run once:
    python seed_admin.py

It will prompt for email and password, then insert:
  1. Organization "REUDE Technologies"
  2. A Profile with role=super_admin, profile_status=approved
"""

import uuid
import sys
import getpass

import bcrypt
from dotenv import load_dotenv

load_dotenv()

from models import SessionLocal, Organization, Profile, init_db


def main():
    # Ensure tables exist
    init_db()

    db = SessionLocal()

    print("\n=== REUDE Local Auth — Seed Super Admin ===\n")

    email = input("Super admin email: ").strip().lower()
    if not email:
        print("Email cannot be empty.")
        sys.exit(1)

    password = getpass.getpass("Password (min 6 chars): ")
    if len(password) < 6:
        print("Password must be at least 6 characters.")
        sys.exit(1)

    password_confirm = getpass.getpass("Confirm password: ")
    if password != password_confirm:
        print("Passwords do not match.")
        sys.exit(1)

    try:
        # 1. Find or create REUDE Technologies org
        org = db.query(Organization).filter(Organization.name == "REUDE Technologies").first()
        if org is None:
            org = Organization(
                id=uuid.uuid4(),
                name="REUDE Technologies",
                max_users=50,
            )
            db.add(org)
            db.flush()
            print(f"  ✅ Created organization: REUDE Technologies (id={org.id})")
        else:
            print(f"  ℹ️  Organization already exists: REUDE Technologies (id={org.id})")

        # 2. Check if email already exists
        existing = db.query(Profile).filter(Profile.email == email).first()
        if existing:
            print(f"  ⚠️  Profile with email '{email}' already exists (id={existing.id}, role={existing.role}).")
            print("     No changes made.")
            db.close()
            return

        # 3. Create super admin profile
        hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        profile = Profile(
            id=uuid.uuid4(),
            email=email,
            password_hash=hashed,
            full_name="Super Admin",
            role="super_admin",
            organization_id=org.id,
            is_active=True,
            profile_status="approved",
            email_verified=True,
        )
        db.add(profile)
        db.commit()

        print(f"  ✅ Created super admin profile:")
        print(f"     Email:  {email}")
        print(f"     Role:   super_admin")
        print(f"     Status: approved")
        print(f"     Org:    REUDE Technologies")
        print(f"     ID:     {profile.id}")
        print("\n  You can now log in to the Streamlit app with these credentials.\n")

    except Exception as exc:
        db.rollback()
        print(f"\n  ❌ Error: {exc}")
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
