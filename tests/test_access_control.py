from src.security.access_control import (
    AccessControl
)

def main():

    roles = [
        "employee",
        "manager",
        "hr",
        "admin",
        "unknown"
    ]

    for role in roles:

        permissions = (
            AccessControl
            .get_allowed_levels(
                role
            )
        )

        print(
            f"{role}: {permissions}"
        )

if __name__ == "__main__":
    main()