from src.security.user_context import (
    UserContext
)

def main():

    user = UserContext(
        user_id="123",
        role="employee",
        departments=["engineering"]
    )

    print(user)

if __name__ == "__main__":
    main()