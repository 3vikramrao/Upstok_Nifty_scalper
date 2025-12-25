import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)
print("Config loaded:", config)
print("CRT config:", config['strategies']['crt_hourly'])