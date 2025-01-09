import json
import datetime
import os

class ResultsLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    def log_result(self, original_path, enhanced_path, metrics, domain_info):
        """Log enhancement results"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_entry = {
            "timestamp": timestamp,
            "original_image": original_path,
            "enhanced_image": enhanced_path,
            "metrics": metrics,
            "domain_info": domain_info
        }
        
        log_file = os.path.join(self.log_dir, f"enhancement_log_{timestamp}.json")
        with open(log_file, 'w') as f:
            json.dump(log_entry, f, indent=4)