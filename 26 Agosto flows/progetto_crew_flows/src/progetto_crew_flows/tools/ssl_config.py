"""
SSL Configuration and HTTP client setup to handle certificate verification issues
"""
import ssl
import httpx
import certifi
from typing import Optional

def create_safe_ssl_context() -> ssl.SSLContext:
    """Create an SSL context that handles certificate verification more gracefully"""
    try:
        # Try to create a context with system certificates
        context = ssl.create_default_context(cafile=certifi.where())
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        return context
    except Exception as e:
        print(f"⚠️ Warning: Could not create SSL context with certificates: {e}")
        # Fallback to unverified context (not recommended for production)
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context

def create_http_client(timeout: int = 30) -> httpx.Client:
    """Create an HTTP client with proper SSL configuration"""
    try:
        ssl_context = create_safe_ssl_context()
        return httpx.Client(
            timeout=timeout,
            verify=ssl_context,
            follow_redirects=True
        )
    except Exception as e:
        print(f"⚠️ Warning: Could not create HTTP client with SSL: {e}")
        # Fallback to basic client
        return httpx.Client(
            timeout=timeout,
            verify=False,  # Not recommended for production
            follow_redirects=True
        )

def configure_azure_openai_ssl():
    """Configure SSL settings for Azure OpenAI connections"""
    import os
    
    # Set environment variables to help with SSL issues
    os.environ.setdefault("PYTHONHTTPSVERIFY", "1")
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    
    print("✅ SSL configuration applied for Azure OpenAI")
