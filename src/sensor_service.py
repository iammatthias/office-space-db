#!/usr/bin/python
# -*- coding:utf-8 -*-

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import time
import smbus
from datetime import datetime
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

# Initialize sensors
try:
    bme280 = BME280.BME280()
    bme280.get_calib_param()
    light = TSL2591.TSL2591()
    uv = LTR390.LTR390()
    sgp = SGP40.SGP40()
    print("Sensors initialized successfully.")
except Exception as e:
    print(f"Error initializing sensors: {e}")
    sys.exit(1)

# Identify MPU/ICM device
try:
    time.sleep(0.1)  # Allow I2C bus to stabilize
    device_id1 = bus.read_byte_data(ICM_SLAVE_ADDRESS, ICM_ADD_WIA)
    device_id2 = bus.read_byte_data(ICM_SLAVE_ADDRESS, MPU_ADD_WIA)
    print(f"Detected IDs: device_id1=0x{device_id1:02X}, device_id2=0x{device_id2:02X}")

    if device_id1 == ICM_VAL_WIA:
        mpu = ICM20948.ICM20948()
        print("ICM20948 detected at I2C address: 0x68")
    elif device_id2 == MPU_VAL_WIA:
        mpu = MPU925x.MPU925x()
        print("MPU925x detected at I2C address: 0x68")
    else:
        print("No compatible MPU/ICM device found.")
        sys.exit(1)
except Exception as e:
    print(f"Error identifying MPU/ICM device: {e}")
    sys.exit(1)

print("Starting data collection... Press Ctrl+C to exit.")

try:
    while True:
        try:
            # Read from BME280
            bme = bme280.readData()
            pressure = round(bme[0], 2)
            temp = round(bme[1], 2)
            hum = round(bme[2], 2)
        except Exception as e:
            print(f"Error reading BME280: {e}")
            pressure, temp, hum = None, None, None

        try:
            # Read from TSL2591
            lux_val = round(light.Lux(), 2)
        except Exception as e:
            print(f"Error reading TSL2591: {e}")
            lux_val = None

        try:
            # Read from LTR390 (UV sensor)
            uvs = float(uv.UVS())
        except Exception as e:
            print(f"Error reading LTR390: {e}")
            uvs = None

        try:
            # Read from SGP40 (VOC sensor)
            gas_val = round(float(sgp.raw()), 2)
        except Exception as e:
            print(f"Error reading SGP40: {e}")
            gas_val = None

        try:
            # Read from ICM/MPU
            icm = mpu.getdata()  # [roll, pitch, yaw, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z]
        except Exception as e:
            print(f"Error reading MPU/ICM: {e}")
            icm = [None] * 12

        # Print out data
        print("=============================================")
        print(f"pressure : {pressure} hPa")
        print(f"temp     : {temp} ℃")
        print(f"hum      : {hum} %")
        print(f"lux      : {lux_val}")
        print(f"uv       : {uvs}")
        print(f"gas      : {gas_val}")
        print(f"Roll     : {icm[0]}, Pitch: {icm[1]}, Yaw: {icm[2]}")
        print(f"Accel    : X = {icm[3]}, Y = {icm[4]}, Z = {icm[5]}")
        print(f"Gyro     : X = {icm[6]}, Y = {icm[7]}, Z = {icm[8]}")
        print(f"Mag      : X = {icm[9]}, Y = {icm[10]}, Z = {icm[11]}")

        # Insert data into Supabase
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

        # Wait for the specified sample rate
        time.sleep(SAMPLE_RATE)

except KeyboardInterrupt:
    print("Exiting...")
except Exception as e:
    print(f"Unexpected error: {e}")
