"""ALLEDITS error taxonomy. Failures must be recoverable and legible (Principle 9)."""


class AllEditsError(Exception):
    """Base class for all ALLEDITS errors."""
    recoverable = False


class MediaError(AllEditsError):
    recoverable = True


class UnsupportedMediaError(MediaError):
    pass


class ProbeError(MediaError):
    pass


class AnalysisError(AllEditsError):
    recoverable = True


class TimelineValidationError(AllEditsError):
    """Raised when a timeline fails validation. NEVER render an invalid timeline."""
    def __init__(self, issues):
        self.issues = list(issues)
        super().__init__(f"Timeline failed validation with {len(self.issues)} issue(s): "
                         + "; ".join(str(i) for i in self.issues[:5]))


class RenderError(AllEditsError):
    recoverable = True


class ProviderError(AllEditsError):
    recoverable = True


class ProviderUnavailable(ProviderError):
    """A model provider is not configured/reachable. Callers must degrade explicitly,
    never silently pretend an LLM ran."""
    pass
