"""
This script is a workaround for a common SSL certificate issue on macOS
that can prevent Python from downloading files over HTTPS.
It installs the `certifi` package and ensures Python's SSL context
uses the certificates from it.
"""

import os
import ssl
import stat
import subprocess
import sys

STAT_0o775 = ( stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
             | stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP
             | stat.S_IROTH |                stat.S_IXOTH )

def main():
    """
    Installs and configures SSL certificates for the current Python environment.
    """
    print("--- Running SSL Certificate Fix ---")
    
    openssl_dir, openssl_cafile = os.path.split(
        ssl.get_default_verify_paths().openssl_cafile)

    print(f"Checking for certificate installation in: {openssl_dir}")

    # Install certifi if it's not already installed
    try:
        import certifi
        print("certifi is already installed.")
    except ImportError:
        print("certifi not found. Installing now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "certifi"])
        import certifi
        print("certifi installed successfully.")

    # Create the target directory if it doesn't exist
    if not os.path.exists(openssl_dir):
        print(f"Creating directory: {openssl_dir}")
        os.makedirs(openssl_dir)

    # Symlink the certifi certificates to the location OpenSSL expects
    cafile = os.path.join(openssl_dir, openssl_cafile)
    if not os.path.exists(cafile):
        print(f"Linking certifi certificates to {cafile}")
        os.symlink(certifi.where(), cafile)
        print("Certificates linked successfully.")
    else:
        print("Certificates are already linked.")

    # Set permissions for the directory
    if os.path.exists(openssl_dir):
        print(f"Setting permissions for {openssl_dir}")
        os.chmod(openssl_dir, STAT_0o775)
    
    print("\n--- SSL Certificate Fix Complete ---")
    print("You can now re-run the `run_processing.py` script.")

if __name__ == "__main__":
    main()
