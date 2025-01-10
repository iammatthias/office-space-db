#!/usr/bin/python
# -*- coding:utf-8 -*-

import sys
import os
import time
from datetime import datetime
import smbus
import subprocess
from supabase import create_client, Client
from python import ICM20948, MPU925x, BME280, LTR390, TSL2591, SGP40
from config.config import SUPABASE_URL, SUPABASE_KEY, SAMPLE_RATE

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Constants for I2C devices
MPU_VAL_WIA = 0x71  # MPU925x ID
MPU_ADD_WIA = 0x75  # MPU925x WHO_AM_I register
ICM_VAL_WIA = 0xEA  # ICM20948 ID
ICM_ADD_WIA = 0x00  # ICM20948 WHO_AM_I register
ICM_SLAVE_ADDRESS = 0x68  # I2C address for both MPU925x and ICM20948

# Other sensor I2C addresses
TSL2591_I2C_ADDRESS = 0x29
LTR390_I2C_ADDRESS = 0x53
SGP40_I2C_ADDRESS = 0x59
BME280_I2C_ADDRESS = 0x76

# Initialize the I2C bus
bus = smbus.SMBus(1)

# Retry logic for sensor initialization
def initialize_sensor(sensor_class, name):
    try:
        return sensor_class()
    except Exception as e:
        print(f"Error initializing {name}: {e}")
        return None

# Restart the Raspberry Pi
def reboot_system():
    print("Rebooting system due to sensor failure...")
    subprocess.run(["sudo", "reboot"], check=True)

# Initialize sensors with retry logic
bme280 = initialize_sensor(BME280.BME280, "BME280")
light = initialize_sensor(TSL2591.TSL2591, "TSL2591")
uv = initialize_sensor(LTR390.LTR390, "LTR390")
sgp = initialize_sensor(SGP40.SGP40, "SGP40")

# Identify MPU/ICM device
try:
    time.sleep(0.1)  # Allow I2C bus to stabilize
    device_id1 = bus.read_byte_data(ICM_SLAVE_ADDRESS, ICM_ADD_WIA)
    device_id2 = bus.read_byte_data(ICM_SLAVE_ADDRESS, MPU_ADD_WIA)
    if device_id1 == ICM_VAL_WIA:
        mpu = ICM20948.ICM20948()
    elif device_id2 == MPU_VAL_WIA:
        mpu = MPU925x.MPU925x()
    else:
        print("No compatible MPU/ICM device found.")
        sys.exit(1)
except Exception as e:
    print(f"Error identifying MPU/ICM device: {e}")
    sys.exit(1)

# Initialize failure counters
sensor_failures = {
    "BME280": 0,
    "TSL2591": 0,
    "LTR390": 0,
    "SGP40": 0,
    "ICM": 0
}

MAX_FAILURES = 5  # Threshold for triggering a reboot

# Data collection loop
try:
    while True:
        try:
            bme = bme280.readData() if bme280 else [None, None, None]
            pressure, temp, hum = round(bme[0], 2), round(bme[1], 2), round(bme[2], 2)
            sensor_failures["BME280"] = 0
        except Exception as e:
            print(f"Error reading BME280: {e}")
            sensor_failures["BME280"] += 1
            if sensor_failures["BME280"] >= MAX_FAILURES:
                reboot_system()

        try:
            lux_val = round(light.Lux(), 2) if light else None
            sensor_failures["TSL2591"] = 0
        except Exception as e:
            print(f"Error reading TSL2591: {e}")
            sensor_failures["TSL2591"] += 1
            if sensor_failures["TSL2591"] >= MAX_FAILURES:
                reboot_system()

        try:
            uvs = float(uv.UVS()) if uv else None
            sensor_failures["LTR390"] = 0
        except Exception as e:
            print(f"Error reading LTR390: {e}")
            sensor_failures["LTR390"] += 1
            if sensor_failures["LTR390"] >= MAX_FAILURES:
                reboot_system()

        try:
            gas_val = round(float(sgp.raw()), 2) if sgp else None
            sensor_failures["SGP40"] = 0
        except Exception as e:
            print(f"Error reading SGP40: {e}")
            sensor_failures["SGP40"] += 1
            if sensor_failures["SGP40"] >= MAX_FAILURES:
                reboot_system()

        try:
            icm = mpu.getdata() if mpu else [None] * 12
            sensor_failures["ICM"] = 0
        except Exception as e:
            print(f"Error reading MPU/ICM: {e}")
            sensor_failures["ICM"] += 1
            if sensor_failures["ICM"] >= MAX_FAILURES:
                reboot_system()

        # Prepare data for insertion
        data_to_insert = {
            "time": datetime.utcnow().isoformat(),
            "pressure": pressure,
            "temp": temp,
            "hum": hum,
            "lux": lux_val,
            "uv": uvs,
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
            "mag_z": icm[11]
        }

        try:
            supabase.table("environmental_data").insert(data_to_insert).execute()
        except Exception as e:
            print(f"Error inserting data into Supabase: {e}")

        time.sleep(SAMPLE_RATE)

except KeyboardInterrupt:
    print("Exiting...")
except Exception as e:
    print(f"Unexpected error: {e}")
