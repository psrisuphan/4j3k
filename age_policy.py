"""Age-dependent policy rules for hate-speech blocking decisions."""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AgePolicy:
    """Simple container for age-specific thresholds."""

    max_age: int
    threshold: float
    allow_unblock: bool


_DEFAULT_POLICIES = (
    AgePolicy(max_age=12, threshold=0.25, allow_unblock=False),
    AgePolicy(max_age=17, threshold=0.35, allow_unblock=True),
    AgePolicy(max_age=200, threshold=0.55, allow_unblock=True),
)


class AgePolicyResolver:
    """Maps a user-provided age to an appropriate policy bucket."""

    def __init__(self, policies: Optional[tuple[AgePolicy, ...]] = None) -> None:
        self._policies = policies or _DEFAULT_POLICIES

    def resolve(self, age: Optional[int]) -> AgePolicy:
        """Return the policy matching ``age``; fall back to the strictest option."""

        if age is None:
            return self._policies[0]

        try:
            integer_age = int(age)
        except (TypeError, ValueError):
            return self._policies[0]

        if integer_age < 0:
            return self._policies[0]

        for policy in self._policies:
            if integer_age <= policy.max_age:
                return policy

        return self._policies[-1]


_resolver = AgePolicyResolver()


def resolve_policy(age: Optional[int]) -> AgePolicy:
    """Return the default policy for ``age`` using a shared resolver instance."""

    return _resolver.resolve(age)
