from __future__ import annotations


#Something broke that might work on a second try: a timeout, a network blip, redis or
#postgres briefly unavailable. Worth retrying.
class TransientIngestionError(Exception):
    pass


#Something is wrong with the data or the request itself: unknown ticker, malformed
#filing, a form type that doesn't exist. Retrying gets the same answer, so don't bother,
#send it straight to the dead letter queue.
class PermanentIngestionError(Exception):
    pass
