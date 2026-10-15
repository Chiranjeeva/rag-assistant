from dataclasses import dataclass, field

@dataclass
class UserContext:
    """
    Represents the current user.

    This will eventually come from:
    - Azure AD
    - Entra ID
    - Streamlit authentication
    - Corporate SSO
    """

    user_id: str
    role: str
    departments: list[str] = field(
        default_factory=list
    )
    groups: list[str] = field(
        default_factory=list
    )