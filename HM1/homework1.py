import json
import matplotlib.pyplot as plt
import numpy as np


def count_prices_above_threshold(data, threshold):
    return sum(1 for item in data if item["price"] > threshold)

# ====================== SUV SECTION ======================
with open("suvs.json", "r") as f:
    data = json.load(f)

# #1
min_price_car = min(data, key=lambda x: x["price"])
max_price_car = max(data, key=lambda x: x["price"])

print('=== SUV Section ===')
print("\n== Pret minim ==")
print(min_price_car)

print("\n== Pret  maxim ==")
print(max_price_car)

# #2 Plot 1: mileage vs car price
mileage = [car["mileage"] for car in data]
prices = [car["price"] for car in data]

plt.figure(figsize=(7, 5))
plt.scatter(mileage, prices)
plt.xlabel("Mileage")
plt.ylabel("Price")
plt.title("Mileage vs Price (SUV)")
plt.grid(True)
plt.show()

suv_prices = [car["price"] for car in data]

mean_price_suv = np.mean(suv_prices)
std_price_suv = np.std(suv_prices)

print("\n== Statistici SUV ==")
print("Mean:", mean_price_suv)
print("Standard deviation:", std_price_suv)

# #3
threshold = 30000   # example
count_suv = count_prices_above_threshold(data, threshold)
print(f"\nNumar SUV-uri cu pretul peste {threshold}: {count_suv}")

# #4
# ---- Boxplot ----
plt.figure(figsize=(6, 4))
plt.boxplot(suv_prices)
plt.title("SUV Price Distribution - Boxplot")
plt.ylabel("Price")
plt.grid(True)
plt.show()

# ---- Violin plot ----
plt.figure(figsize=(6, 4))
plt.violinplot(suv_prices, showmeans=True)
plt.title("SUV Price Distribution - Violin Plot")
plt.ylabel("Price")
plt.grid(True)
plt.show()

# ====================== APARTMENTS SECTION ======================
with open("apartments.json", "r") as f:
    data = json.load(f)

# #1
min_price_apartment = min(data, key=lambda x: x["price"])
max_price_apartment = max(data, key=lambda x: x["price"])

print('=== Apartments Section ===')
print("\n== Pret minim ==")
print(min_price_apartment)

print("\n== Pret maxim ==")
print(max_price_apartment)

# #2 Plot 2: surface vs apartment price
surface = [apt["surface"] for apt in data]
prices = [apt["price"] for apt in data]

plt.figure(figsize=(7, 5))
plt.scatter(surface, prices)
plt.xlabel("Surface (m2)")
plt.ylabel("Price")
plt.title("Surface vs Price (Apartments)")
plt.grid(True)
plt.show()

# #3
threshold = 100000  # example
count_apartments = count_prices_above_threshold(data, threshold)
print(f"\nNumar apartamente cu pretul peste {threshold}: {count_apartments}")

apartment_prices = [apt["price"] for apt in data]

mean_price_apt = np.mean(apartment_prices)
std_price_apt = np.std(apartment_prices)

print("\n== Statistici Apartamente ==")
print("Mean:", mean_price_apt)
print("Standard deviation:", std_price_apt)

# ---- Boxplot ----
plt.figure(figsize=(6, 4))
plt.boxplot(apartment_prices)
plt.title("Apartment Price Distribution - Boxplot")
plt.ylabel("Price")
plt.grid(True)
plt.show()

# ---- Violin plot ----
plt.figure(figsize=(6, 4))
plt.violinplot(apartment_prices, showmeans=True)
plt.title("Apartment Price Distribution - Violin Plot")
plt.ylabel("Price")
plt.grid(True)
plt.show()