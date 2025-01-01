#!/usr/bin/env python3

import time
from datetime import datetime
import json
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import SUPABASE_URL, SUPABASE_KEY, SAMPLE_RATE
from supabase import create_client
from typing import Dict, Any

# Allowed sensor IDs based on RLS policy
ALLOWED_SENSOR_IDS = {"icm20948", "bme280", "ltr390", "tsl25911", "sgp40"}


class MockSensor:
    def read_accelerometer(self): return (0, 0, 0)
    def read_gyroscope(self): return (0, 0, 0)
    def read_magnetometer(self): return (0, 0, 0)
    def read_temperature(self): return 20
    def read_pressure(self): return 1000
    def read_humidity(self): return 50
    def read_uv(self): return 0
    def read_lux(self): return 500
    def read_voc(self): return 100
    def get_calib_param(self): pass


class SensorService:
    def __init__(self):
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("Missing Supabase credentials")
        
        self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.mock = MockSensor()

    def read_sensors(self) -> list:
        timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')  # ISO 8601 format
        readings = []

        try:
            # ICM20948 - 3-axis readings use "values"
            ax, ay, az = self.mock.read_accelerometer()
            gx, gy, gz = self.mock.read_gyroscope()
            mx, my, mz = self.mock.read_magnetometer()
            
            readings.extend([
                {"sensor_id": "icm20948", "type": "accelerometer", "timestamp": timestamp, "values": [{"x": ax, "y": ay, "z": az}]},
                {"sensor_id": "icm20948", "type": "gyroscope", "timestamp": timestamp, "values": [{"x": gx, "y": gy, "z": gz}]},
                {"sensor_id": "icm20948", "type": "magnetometer", "timestamp": timestamp, "values": [{"x": mx, "y": my, "z": mz}]}
            ])

            # Single value readings as one-element arrays
            readings.extend([
                {"sensor_id": "bme280", "type": "temperature", "timestamp": timestamp, "values": [self.mock.read_temperature()]},
                {"sensor_id": "bme280", "type": "pressure", "timestamp": timestamp, "values": [self.mock.read_pressure()]},
                {"sensor_id": "bme280", "type": "humidity", "timestamp": timestamp, "values": [self.mock.read_humidity()]},
                {"sensor_id": "ltr390", "type": "uv", "timestamp": timestamp, "values": [self.mock.read_uv()]},
                {"sensor_id": "tsl25911", "type": "light", "timestamp": timestamp, "values": [self.mock.read_lux()]},
                {"sensor_id": "sgp40", "type": "voc", "timestamp": timestamp, "values": [self.mock.read_voc()]}
            ])
        except Exception as e:
            print(f"Error reading sensors: {e}")
        
        return readings
    
    def store_data(self, readings: list) -> None:
        if not readings:
            return

        # Validate readings for RLS compliance
        validated_readings = [
            reading for reading in readings if reading["sensor_id"] in ALLOWED_SENSOR_IDS
        ]

        try:
            if validated_readings:
                response = self.supabase.table("sensor_readings").insert(validated_readings).execute()
                print(f"Insert Response: {response}")  # Log full response
                if response.get("status_code") != 201:
                    print(f"Insert error: {response.get('data')}")
            else:
                print("No valid sensor data to store.")
        except Exception as e:
            print(f"Error storing data: {e}")

    def run(self):
        print(f"Starting sensor service, sampling every {SAMPLE_RATE} seconds")
        while True:
            readings = self.read_sensors()
            self.store_data(readings)
            time.sleep(SAMPLE_RATE)


if __name__ == "__main__":
    service = SensorService()
    service.run()
