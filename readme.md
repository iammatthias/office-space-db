# office---space.db

Python and metadata for the sensors that are used to track the enviromental conditions of my office space.

## Sensors
- ICM20948: 9-DoF Motion Tracking
- BME280: Temperature, Humidity, Pressure
- LTR390: UV Light
- TSL25911: Ambient Light
- SGP40: VOC Air Quality

## Setup
```bash
# Install dependencies
pip3 install -r requirements.txt

# Configure environment
cp config/.env.example config/.env
# Edit config/.env with your Supabase credentials

# Install service (Raspberry Pi)
sudo cp systemd/sensor-service.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable sensor-service
sudo systemctl start sensor-service
```

## Development
```bash
python3 src/sensor_service.py
```

## Monitoring
```bash
# View logs
sudo journalctl -u sensor-service -f

# Check status
sudo systemctl status sensor-service
```