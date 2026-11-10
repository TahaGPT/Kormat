"""Label space: each criterion type keeps its own four expert categories (never collapsed to yes/no)."""

LABELS: dict[str, tuple[str, ...]] = {
    "inclusion": ("included", "not included", "not enough information", "not applicable"),
    "exclusion": ("excluded", "not excluded", "not enough information", "not applicable"),
}
ALL_LABELS: tuple[str, ...] = (
    "included", "not included", "excluded", "not excluded", "not enough information", "not applicable",
)
INVALID = "INVALID"
