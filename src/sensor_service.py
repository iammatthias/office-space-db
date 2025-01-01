#!/usr/bin/python
# -*- coding:utf-8 -*-

import time
import os
import smbus
import psycopg2

from python import ICM20948  # Gyroscope/Acceleration/Magnetometer
from python import MPU925x   # Gyroscope/Acceleration/Magnetometer
from python import BME280    # Atmospheric Pressure/Temperature/Humidity
from python import LTR390    # UV
from python import TSL2591   # Light
from python import SGP40     # VOC

from PIL import Image, ImageDraw, ImageFont

MPU_VAL_WIA       = 0x71
MPU_ADD_WIA       = 0x75
ICM_VAL_WIA       = 0xEA
ICM_ADD_WIA       = 0x00
ICM_SLAVE_ADDRESS = 0x68

bus = smbus.SMBus(1)

bme280 = BME280.BME280()
bme280.get_calib_param()

light = TSL2591.TSL2591()
uv = LTR390.LTR390()
sgp = SGP40.SGP40()

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

SUPABASE_DB_URL = os.environ.get("SUPABASE_DB_URL", "postgresql://YOUR_USER:YOUR_PASSWORD@YOUR_HOST:5432/YOUR_DB?sslmode=require")

try:
    conn = psycopg2.connect(SUPABASE_DB_URL)
    conn.autocommit = True
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS environmental_data (
            time        TIMESTAMPTZ NOT NULL,
            pressure    DOUBLE PRECISION,
            temp        DOUBLE PRECISION,
            hum         DOUBLE PRECISION,
            lux         DOUBLE PRECISION,
            uv          DOUBLE PRECISION,
            gas         DOUBLE PRECISION,
            roll        DOUBLE PRECISION,
            pitch       DOUBLE PRECISION,
            yaw         DOUBLE PRECISION,
            accel_x     INTEGER,
            accel_y     INTEGER,
            accel_z     INTEGER,
            gyro_x      INTEGER,
            gyro_y      INTEGER,
            gyro_z      INTEGER,
            mag_x       INTEGER,
            mag_y       INTEGER,
            mag_z       INTEGER
        );
    """)

    cursor.execute("""
        SELECT create_hypertable('environmental_data', 'time', if_not_exists => TRUE);
    """)

    print("Starting data collection... Press Ctrl+C to exit.")
    while True:
        bme = bme280.readData()
        pressure = round(bme[0], 2)
        temp = round(bme[1], 2)
        hum = round(bme[2], 2)
        lux_val = round(light.Lux(), 2)
        uvs = uv.UVS()
        gas_val = round(sgp.raw(), 2)
        icm = mpu.getdata()  # [roll, pitch, yaw, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z]

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

        cursor.execute("""
            INSERT INTO environmental_data (
                time,
                pressure,
                temp,
                hum,
                lux,
                uv,
                gas,
                roll,
                pitch,
                yaw,
                accel_x,
                accel_y,
                accel_z,
                gyro_x,
                gyro_y,
                gyro_z,
                mag_x,
                mag_y,
                mag_z
            )
            VALUES (
                NOW(),
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s
            );
        """, (
            pressure,
            temp,
            hum,
            lux_val,
            float(uvs),
            gas_val,
            icm[0],
            icm[1],
            icm[2],
            icm[3],
            icm[4],
            icm[5],
            icm[6],
            icm[7],
            icm[8],
            icm[9],
            icm[10],
            icm[11]
        ))

        time.sleep(60)

except KeyboardInterrupt:
    print("Exiting...")
except Exception as e:
    print(f"Error: {e}")
finally:
    if 'cursor' in globals():
        cursor.close()
    if 'conn' in globals():
        conn.close()
