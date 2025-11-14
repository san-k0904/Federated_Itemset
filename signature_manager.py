# signature_manager.py
"""
Signature Manager using Ed25519.

Provides:
- generate_keypair()
- sign(message_bytes, private_key)
- verify(message_bytes, signature, public_key)
- serialize/deserialize public keys (raw bytes) for distribution
"""

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives import serialization

def generate_keypair():
    """
    Returns: (private_key_obj, public_key_obj)
    """
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()
    return priv, pub

def private_key_bytes(private_key):
    """Serialize private key to bytes (PEM). Use only if you must persist; in-memory is safer."""
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

def public_key_bytes(public_key):
    """Serialize public key to bytes (PEM)"""
    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

def load_public_key(pub_bytes):
    return serialization.load_pem_public_key(pub_bytes)

def load_private_key(priv_bytes):
    return serialization.load_pem_private_key(priv_bytes, password=None)

def sign(message: bytes, private_key) -> bytes:
    """
    Sign message bytes using Ed25519 private key object.
    Returns signature bytes.
    """
    return private_key.sign(message)

def verify(message: bytes, signature: bytes, public_key) -> bool:
    """
    Verify signature. Raises InvalidSignature if invalid; caller may catch.
    """
    public_key.verify(signature, message)
    return True
