class AccessControl:
    """
    Maps user roles to permitted access levels.
    """

    ROLE_MAPPING = {

        "employee": [
            "public"
        ],

        "manager": [
            "public",
            "manager"
        ],

        "hr": [
            "public",
            "manager",
            "hr"
        ],

        "admin": [
            "public",
            "manager",
            "hr",
            "admin"
        ],
    }

    @classmethod
    def get_allowed_levels(
        cls,
        role: str,
    ) -> list[str]:

        return cls.ROLE_MAPPING.get(
            role,
            ["public"]
        )