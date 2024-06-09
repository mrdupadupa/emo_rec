import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

# Next 2 lines are needed to specify the path to your geckodriver
geckodriver_path = "/snap/bin/geckodriver"
driver_service = webdriver.FirefoxService(executable_path=geckodriver_path)

driver = webdriver.Firefox(service=driver_service)
# browser = webdriver.Firefox() # Originial Example

driver.get("http://192.168.194.2:9966/")
#driver.find_element(By.PARTIAL_LINK_TEXT, 'Seed').send_keys("123");

seed_input = driver.find_element(By.CSS_SELECTOR, "input[type='text']")

# Clear the current seed value (if any)
seed_input.clear()

# Enter the new seed value
new_seed_value = "123"  # Replace with the seed value you want
seed_input.send_keys("123")

time.sleep(2)

seed_input.send_keys("278")

# Trigger the change event (if necessary)
seed_input.send_keys(Keys.ENTER)  # You might use Keys.RETURN or other keys as appropriate

# Optionally, you can interact with other GUI elements or buttons as needed using Selenium
