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

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Constants for I2C devices
ICM_SLAVE_ADDRESS = 0x68
MPU_VAL_WIA = 0x71
ICM_VAL_WIA = 0xEA
MPU_ADD_WIA = 0x75
ICM_ADD_WIA = 0x00

# Initialize the I2C bus
bus = smbus.SMBus(1)

# Helper function to initialize sensors with error handling
def initialize_sensor(sensor_class, name):
    try:
        return sensor_class()
    except Exception as e:
        print(f"Error initializing {name}: {e}")
        return None

# Initialize sensors
bme280 = initialize_sensor(BME280.BME280, "BME280")
if bme280:
    try:
        bme280.get_calib_param()
    except Exception as e:
        print(f"Error calibrating BME280: {e}")
        bme280 = None

light = initialize_sensor(TSL2591.TSL2591, "TSL2591")
uv = initialize_sensor(LTR390.LTR390, "LTR390")
sgp = initialize_sensor(SGP40.SGP40, "SGP40")

# Identify MPU/ICM device
try:
    time.sleep(0.1)
    device_id1 = bus.read_byte_data(ICM_SLAVE_ADDRESS, ICM_ADD_WIA)
    device_id2 = bus.read_byte_data(ICM_SLAVE_ADDRESS, MPU_ADD_WIA)
    print(f"Detected IDs: device_id1=0x{device_id1:02X}, device_id2=0x{device_id2:02X}")

    if device_id1 == ICM_VAL_WIA:
        mpu = ICM20948.ICM20948()
        print("ICM20948 detected.")
    elif device_id2 == MPU_VAL_WIA:
        mpu = MPU925x.MPU925x()
        print("MPU925x detected.")
    else:
        print("No compatible MPU/ICM device found.")
        sys.exit(1)
except Exception as e:
    print(f"Error identifying MPU/ICM device: {e}")
    sys.exit(1)

print("Starting data collection... Press Ctrl+C to exit.")

try:
    while True:
        # Default sensor readings
        pressure, temp, hum = None, None, None
        lux_val, uvs, gas_val = None, None, None
        icm = [None] * 12

        try:
            if bme280:
                bme = bme280.readData()
                if len(bme) == 3:  # Ensure valid data length
                    pressure, temp, hum = map(lambda x: round(x, 2), bme)
                else:
                    raise ValueError("Incomplete data from BME280.")
        except Exception as e:
            print(f"Error reading BME280: {e}")

        try:
            if light:
                lux_val = round(light.Lux(), 2)
        except Exception as e:
            print(f"Error reading TSL2591: {e}")

        try:
            if uv:
                uvs = float(uv.UVS())
        except Exception as e:
            print(f"Error reading LTR390: {e}")

        try:
            if sgp:
                gas_val = round(float(sgp.raw()), 2)
        except Exception as e:
            print(f"Error reading SGP40: {e}")

        try:
            if mpu:
                icm = mpu.getdata()
        except Exception as e:
            print(f"Error reading MPU/ICM: {e}")

        # Print sensor readings
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

        # Sleep for the sample rate interval
        time.sleep(SAMPLE_RATE)

except KeyboardInterrupt:
    print("Exiting...")
except Exception as e:
    print(f"Unexpected error: {e}")
