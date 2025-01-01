#!/usr/bin/python
# -*- coding:utf-8 -*-

import time
import os
import smbus
from dotenv import load_dotenv

# Supabase Python client
from supabase import create_client, Client

from python import ICM20948  # Gyroscope/Acceleration/Magnetometer
from python import MPU925x   # Gyroscope/Acceleration/Magnetometer
from python import BME280    # Atmospheric Pressure/Temperature/Humidity
from python import LTR390    # UV
from python import TSL2591   # Light
from python import SGP40     # VOC

# Optional display libraries (if needed)
from PIL import Image, ImageDraw, ImageFont

MPU_VAL_WIA       = 0x71
MPU_ADD_WIA       = 0x75
ICM_VAL_WIA       = 0xEA
ICM_ADD_WIA       = 0x00
ICM_SLAVE_ADDRESS = 0x68

bus = smbus.SMBus(1)

# Initialize sensors
bme280 = BME280.BME280()
bme280.get_calib_param()
light = TSL2591.TSL2591()
uv    = LTR390.LTR390()
sgp   = SGP40.SGP40()

device_id1 = bus.read_byte_data(ICM_SLAVE_ADDRESS, ICM_ADD_WIA)
device_id2 = bus.read_byte_data(ICM_SLAVE_ADDRESS, MPU_ADD_WIA)

if device_id1 == ICM_VAL_WIA:
    mpu = ICM20948.ICM20948()
    print("ICM20948 9-DOF I2C address: 0x68")
elif device_id2 == MPU_VAL_WIA:
    mpu = MPU925x.MPU925x()
    print("MPU925x 9-DOF I2C address: 0x68")

print("TSL2591 Light I2C address: 0x29")
print("LTR390 UV I2C address:     0x53")
print("SGP40 VOC I2C address:     0x59")
print("BME280 T&H I2C address:    0x76")

# -----------------------------------------------------------------------------
# Load environment variables from .env
# -----------------------------------------------------------------------------
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../config/.env'))

# -----------------------------------------------------------------------------
# Set up the Supabase client
# Replace with your actual values or rely on environment variables.
# -----------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SAMPLE_RATE  = int(os.getenv("SAMPLE_RATE", 60))

print("SUPABASE_URL:", SUPABASE_URL)
print("SUPABASE_KEY:", SUPABASE_KEY)

# Initialize the Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print("Starting data collection... Press Ctrl+C to exit.")

try:
    while True:
        # Read from BME280
        bme = bme280.readData()
        pressure = round(bme[0], 2)
        temp     = round(bme[1], 2)
        hum      = round(bme[2], 2)

        # Read from TSL2591
        lux_val  = round(light.Lux(), 2)

        # Read from LTR390 (UV sensor)
        uvs      = uv.UVS()

        # Read from SGP40 (VOC sensor)
        gas_val  = round(sgp.raw(), 2)

        # Read from ICM/MPU
        icm      = mpu.getdata()  # [roll, pitch, yaw, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z]

        # Print out data
        print("=============================================")
        print(f"pressure : {pressure} hPa")
        print(f"temp     : {temp} ℃")
        print(f"hum      : {hum} %")
        print(f"lux      : {lux_val}")
        print(f"uv       : {uvs}")
        print(f"gas      : {gas_val}")
        print(f"Roll     : {icm[0]:.2f}, Pitch: {icm[1]:.2f}, Yaw: {icm[2]:.2f}")
        print(f"Accel    : X = {icm[3]}, Y = {icm[4]}, Z = {icm[5]}")
        print(f"Gyro     : X = {icm[6]}, Y = {icm[7]}, Z = {icm[8]}")
        print(f"Mag      : X = {icm[9]}, Y = {icm[10]}, Z = {icm[11]}")

        # Insert data into Supabase
        data_to_insert = {
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
            "mag_z": icm[11]
        }

        supabase.table("environmental_data").insert(data_to_insert).execute()

        # Wait 60 seconds
        time.sleep(60)

except KeyboardInterrupt:
    print("Exiting...")
except Exception as e:
    print(f"Error: {e}")