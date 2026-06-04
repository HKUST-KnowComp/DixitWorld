x# -*- coding: utf-8 -*-
"""
网络诊断工具
用于检测和解决OpenRouter API连接问题
"""

import requests
import time
import ssl
import socket
from urllib.parse import urlparse

def test_ssl_connection():
    """测试SSL连接"""
    print("🔍 测试SSL连接...")
    try:
        context = ssl.create_default_context()
        with socket.create_connection(("openrouter.ai", 443), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname="openrouter.ai") as ssock:
                print(f"✅ SSL连接成功")
                print(f"   协议版本: {ssock.version()}")
                print(f"   加密套件: {ssock.cipher()}")
                return True
    except Exception as e:
        print(f"❌ SSL连接失败: {e}")
        return False

def test_http_connection():
    """测试HTTP连接"""
    print("\n🔍 测试HTTP连接...")
    try:
        response = requests.get("https://openrouter.ai", timeout=10)
        print(f"✅ HTTP连接成功，状态码: {response.status_code}")
        return True
    except Exception as e:
        print(f"❌ HTTP连接失败: {e}")
        return False

def test_api_endpoint():
    """测试API端点"""
    print("\n🔍 测试API端点...")
    try:
        # 测试一个简单的API调用
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            timeout=15,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )
        print(f"✅ API端点连接成功，状态码: {response.status_code}")
        return True
    except Exception as e:
        print(f"❌ API端点连接失败: {e}")
        return False

def test_with_retry():
    """测试带重试的连接"""
    print("\n🔍 测试带重试的连接...")
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            print(f"   尝试 {attempt + 1}/{max_retries}...")
            response = requests.get(
                "https://openrouter.ai/api/v1/models",
                timeout=15,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
            )
            print(f"✅ 重试连接成功，状态码: {response.status_code}")
            return True
        except Exception as e:
            print(f"   尝试 {attempt + 1} 失败: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"   等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
    
    print(f"❌ 所有重试都失败了")
    return False

def suggest_solutions():
    """建议解决方案"""
    print("\n💡 建议的解决方案:")
    print("1. 检查网络连接是否稳定")
    print("2. 尝试使用VPN或更换网络环境")
    print("3. 检查防火墙设置，确保允许HTTPS连接")
    print("4. 更新Python的requests和urllib3库:")
    print("   pip install --upgrade requests urllib3")
    print("5. 如果问题持续，可以尝试:")
    print("   - 重启网络适配器")
    print("   - 清除DNS缓存")
    print("   - 检查系统时间是否正确")
    print("6. 在代码中增加更长的等待时间")

def main():
    """主函数"""
    print("🌐 OpenRouter API 网络诊断工具")
    print("=" * 50)
    
    # 运行所有测试
    ssl_ok = test_ssl_connection()
    http_ok = test_http_connection()
    api_ok = test_api_endpoint()
    retry_ok = test_with_retry()
    
    print("\n📊 诊断结果:")
    print(f"SSL连接: {'✅' if ssl_ok else '❌'}")
    print(f"HTTP连接: {'✅' if http_ok else '❌'}")
    print(f"API端点: {'✅' if api_ok else '❌'}")
    print(f"重试连接: {'✅' if retry_ok else '❌'}")
    
    if not all([ssl_ok, http_ok, api_ok, retry_ok]):
        suggest_solutions()
    else:
        print("\n🎉 所有测试都通过了！网络连接正常。")

if __name__ == "__main__":
    main()
