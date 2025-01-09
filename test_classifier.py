from models.domain_classifier import DomainClassifier
import os

def test_domain_classifier():
    # Initialize classifier
    classifier = DomainClassifier()
    
    # Test folder path
    test_folder = "test_images"
    
    # Create test folder if it doesn't exist
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
        print(f"Created {test_folder} directory.")
        print("Please add some test images to this folder!")
        return
    
    # Test classification on all images in the folder
    for image_file in os.listdir(test_folder):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(test_folder, image_file)
            
            try:
                domain, confidence = classifier.classify(image_path)
                print(f"\nImage: {image_file}")
                print(f"Predicted Domain: {domain}")
                print(f"Confidence: {confidence:.2%}")
            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")

if __name__ == "__main__":
    test_domain_classifier()