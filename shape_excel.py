from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

# 设置ChromeDriver路径
chrome_driver_path = "/path/to/chromedriver"

# 启动Chrome浏览器
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service)

# 打开目标网页
url = "https://example.com"
driver.get(url)

# 查找所有class为"XqQF9c"的<a>元素
elements = driver.find_elements(By.CLASS_NAME, "XqQF9c")

# 提取并打印每个<a>元素的href属性
for element in elements:
    href = element.get_attribute("href")
    if href:
        print(href)

# 关闭浏览器
driver.quit()
