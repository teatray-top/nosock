import ctypes
import ctypes.wintypes as wt
import sys

CRED_TYPE_GENERIC = 1
CRED_PERSIST_LOCAL_MACHINE = 2
adv = ctypes.windll.advapi32


class CREDENTIAL(ctypes.Structure):
    _fields_ = [("Flags", wt.DWORD), ("Type", wt.DWORD), ("TargetName", wt.LPWSTR), ("Comment", wt.LPWSTR), ("LastWritten", wt.FILETIME), ("CredentialBlobSize", wt.DWORD), ("CredentialBlob", ctypes.POINTER(ctypes.c_byte)), ("Persist", wt.DWORD), ("AttributeCount", wt.DWORD), ("Attributes", ctypes.c_void_p), ("TargetAlias", wt.LPWSTR), ("UserName", wt.LPWSTR)]


def store(target, user, secret):
    blob = secret.encode("utf-16-le")
    buf = (ctypes.c_byte * len(blob)).from_buffer_copy(blob)
    c = CREDENTIAL(Type=CRED_TYPE_GENERIC, TargetName=target, CredentialBlobSize=len(blob), CredentialBlob=ctypes.cast(buf, ctypes.POINTER(ctypes.c_byte)), Persist=CRED_PERSIST_LOCAL_MACHINE, UserName=user)
    if not adv.CredWriteW(ctypes.byref(c), 0):
        raise OSError(ctypes.GetLastError())


def load(target):
    p = ctypes.POINTER(CREDENTIAL)()
    if not adv.CredReadW(target, CRED_TYPE_GENERIC, 0, ctypes.byref(p)):
        raise KeyError(target)
    c = p.contents
    secret = ctypes.string_at(c.CredentialBlob, c.CredentialBlobSize).decode("utf-16-le")
    user = c.UserName
    adv.CredFree(p)
    return user, secret


if __name__ == "__main__":
    import base64
    target, path = sys.argv[1], sys.argv[2]
    txt = base64.b64decode(open(path, encoding="utf-8").read().strip()).decode("utf-8")
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    kv = {lines[i]: lines[i + 1] for i in range(0, len(lines) - 1, 2)}
    store(target, kv["Client Id"], kv["Client Secret"])
    u, s = load(target)
    print(f"stored {target}: id len {len(u)} ({u[:4]}...), secret len {len(s)} ({s[:4]}...), expires {kv.get('만료일')}")
