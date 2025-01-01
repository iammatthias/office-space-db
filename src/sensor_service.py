#!/usr/bin/python
# -*- coding:utf-8 -*-

import sys
import os
import time
import smbus
from datetime import datetime
from supabase import create_client, Client
from python import ICM20948, MPU925x, BME280, LTR390, TSL2591, SGP40
from config.config import SUPABASE_URL, SUPABASE_KEY, SAMPLE_RATE

# Constants for device IDs
MPU_VAL_WIA = 0x71
MPU_ADD_WIA = 0x75
ICM_VAL_WIA = 0xEA
ICM_ADD_WIA = 0x00
ICM_SLAVE_ADDRESS = 0x68

# I2C Bus
bus = smbus.SMBus(1)

# Initialize other sensors
bme280 = BME280.BME280()
bme280.get_calib_param()
light = TSL2591.TSL2591()
uv = LTR390.LTR390()
sgp = SGP40.SGP40()

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Function to initialize MPU sensor
def initialize_mpu():
    try:
        device_id1 = bus.read_byte_data(ICM_SLAVE_ADDRESS, ICM_ADD_WIA)
        device_id2 = bus.read_byte_data(ICM_SLAVE_ADDRESS, MPU_ADD_WIA)
        if device_id1 == ICM_VAL_WIA:
            print("ICM20948 9-DOF I2C address: 0x68")
            return ICM20948.ICM20948()
        elif device_id2 == MPU_VAL_WIA:
            print("MPU925x 9-DOF I2C address: 0x68")
            return MPU925x.MPU925x()
        else:
            print("No supported MPU device found.")
            return None
    except Exception as e:
        print(f"Error initializing MPU: {e}")
        return None

# Initialize MPU sensor
mpu = initialize_mpu()
if not mpu:
    print("Failed to initialize MPU sensor. Exiting...")
    sys.exit(1)

print("TSL2591 Light I2C address: 0x29")
print("LTR390 UV I2C address:     0x53")
print("SGP40 VOC I2C address:     0x59")
print("BME280 T&H I2C address:    0x76")

# Main data collection loop
try:
    while True:
        # Read sensor data
        bme = bme280.readData()
        pressure, temp, hum = round(bme[0], 2), round(bme[1], 2), round(bme[2], 2)
        lux_val = round(light.Lux(), 2)
        uvs = uv.UVS()
        gas_val = round(sgp.raw(), 2)
        icm = mpu.getdata()

        # Print sensor data
        print("=============================================")
        print(f"pressure: {pressure} hPa, temp: {temp} ℃, hum: {hum} %")
        print(f"lux: {lux_val}, uv: {uvs}, gas: {gas_val}")
        print(f"Roll: {icm[0]:.2f}, Pitch: {icm[1]:.2f}, Yaw: {icm[2]:.2f}")
        print(f"Accel: X = {icm[3]}, Y = {icm[4]}, Z = {icm[5]}")
        print(f"Gyro: X = {icm[6]}, Y = {icm[7]}, Z = {icm[8]}")
        print(f"Mag: X = {icm[9]}, Y = {icm[10]}, Z = {icm[11]}")

        # Insert data into Supabase
        data_to_insert = {
            "time": datetime.utcnow().isoformat(),
            "pressure": pressure,
            "temp": temp,
            "hum": hum,
            "lux": lux_val,
            "uv": float(uvs),
            "gas": gas_val,
            "roll": icm[0],
            "pitch": icm[1],
            "yaw": icm[2],
            "accel_x": icm[3],
            "accel_y": icm[4],
            "accel_z": icm[5],
            "gyro_x": icm[6],
            "gyro_y": icm[7],
            "gyro_z": icm[8],
            "mag_x": icm[9],
            "mag_y": icm[10],
            "mag_z": icm[11],
        }
        supabase.table("environmental_data").insert(data_to_insert).execute()

        # Wait for the next sample
        time.sleep(SAMPLE_RATE)

except KeyboardInterrupt:
    print("Exiting...")
except Exception as e:
    print(f"Error: {e}")
