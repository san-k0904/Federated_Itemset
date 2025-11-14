# encryption_manager.py
"""
Encryption Module (AES-256-GCM) for Federated Itemset Mining
Provides:
- AESGCM key generation
- encrypt/decrypt helpers that return/expect nonce + ciphertext
"""

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os

class EncryptionManager:
    """
    AES-256-GCM encryption/decryption wrapper.
    Stores symmetric key bytes in self.key (32 bytes).
    """

    def __init__(self, key: bytes = None):
        """
        If key provided, use it. Otherwise generate a new random 256-bit key.
        """
        if key is not None:
            if not isinstance(key, (bytes, bytearray)) or len(key) != 32:
                raise ValueError("Provided key must be 32 bytes (AES-256).")
            self.key = bytes(key)
        else:
            self.key = AESGCM.generate_key(bit_length=256)

    def get_key(self) -> bytes:
        """Return raw symmetric key bytes for distribution (if you must)."""
        return self.key

    def encrypt(self, plaintext: bytes, associated_data: bytes = None) -> dict:
        """
        Encrypts plaintext bytes and returns a dict envelope with nonce and ciphertext.
        Args:
            plaintext: bytes
            associated_data: optional bytes that are authenticated but not encrypted (AAD)
        Returns:
            {'nonce': bytes, 'ciphertext': bytes}
        """
        aesgcm = AESGCM(self.key)
        nonce = os.urandom(12)  # 96-bit nonce recommended for GCM
        ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data)
        return {'nonce': nonce, 'ciphertext': ciphertext, 'aad': associated_data}

    def decrypt(self, envelope: dict) -> bytes:
        """
        Decrypt envelope produced by encrypt().
        Args:
            envelope: dict with keys 'nonce', 'ciphertext', optionally 'aad'
        Returns:
            plaintext bytes
        Raises:
            cryptography.exceptions.InvalidTag on authentication failure
        """
        aesgcm = AESGCM(self.key)
        nonce = envelope['nonce']
        ciphertext = envelope['ciphertext']
        aad = envelope.get('aad', None)
        plaintext = aesgcm.decrypt(nonce, ciphertext, aad)
        return plaintext
