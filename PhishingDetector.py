import re

class PhishingDetector:
    """
    A class that analyzes email content for signs of phishing attempts.
    """
    
    def __init__(self, email_content, sender_address):
        """
        Initialize the detector with email content and sender address.
        
        Args:
            email_content (str): The body text of the email.
            sender_address (str): The email address of the sender.
        """
        self.email_content = email_content
        self.sender_address = sender_address

    def check_urgent_language(self):
        """
        Check for urgent or pressuring language in the email content.
        
        Returns:
            bool: True if urgent language is detected, otherwise False.
        """
        urgent_phrases = [
            "urgent", 
            "immediate action required",
            "immediate verification needed", 
            "account suspended", 
            "security alert", 
            "verify your account",
            "act now",
            "last chance",
            "important notice",
            "your account will be closed",
            "update your information",
            "confirm your identity",
            "you have won",
            "congratulations"
            "limited time offer"
        ]
        for phrase in urgent_phrases:
            if phrase in self.email_content.lower():
                return True
        return False

    def check_suspicious_links(self):
        """
        Detect suspicious links or URLs in the email content.
        
        Returns:
            bool: True if suspicious links are detected, otherwise False.
        """
        # Regex for detecting shortened URLs and suspicious domains
        suspicious_domains = [
           r"bit\.ly",
            r"tinyurl\.com",
            r"\.tk",
            r"\.xyz",
            r"click here",
            r"http[s]?://"
        ]
        match = any(re.search(pattern, self.email_content) for pattern in suspicious_domains)
        
        # Check for 'click here' with a link
        if "click here" in self.email_content.lower() and re.search(r'https?://', self.email_content):
            match = True
        
        return match

    def analyze_sender_address(self):
        """
        Analyze the sender's email address for signs of spoofing.
        
        Returns:
            bool: True if the email address shows signs of spoofing, otherwise False.
        """
        # Check for common misspellings
        legitimate_domain = "gmail.com",
        parts = self.sender_address.split('@')
        if len(parts) != 2 or not self.sender_address.endswith(legitimate_domain):
            return True
        
        # Check for excessive dots, numbers, or misspelling
        if any(char.isdigit() for char in parts[0]) or "gmal" in parts[1]:
            return True
        
        return False

    def check_sensitive_info_requests(self):
        """
        Identify requests for sensitive personal or financial information.
        
        Returns:
            bool: True if there are requests for sensitive information, otherwise False.
        """
        sensitive_requests = [
            "please provide your password", 
            "send us your credit card number", 
            "social security number", 
            "bank account details"
            "click here to verify your account",
            "update your payment information",
            "confirm your identity",
            "provide your login credentials"
        ]
        
        for request in sensitive_requests:
            if request in self.email_content.lower():
                return True
        return False


def main():
    print("==== Phishing Email Detector ====")
    email_content = input("Enter the email content: ")
    sender_address = input("Enter the sender's email address: ")

    detector = PhishingDetector(email_content, sender_address)

    while True:
        print("\nPHISHING DETECTION OPTIONS")
        print("1. CHECK FOR URGENT LANGUAGE")
        print("2. CHECK FOR SUSPICIOUS LINKS")
        print("3. CHECK FOR SPOOFED SENDER")
        print("4. CHECK FOR SENSITIVE DATA REQUESTS")
        print("5. EXIT")

        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            if detector.check_urgent_language():
                print("WARNING: Urgent language detected - possible phishing attempt.")
            else:
                print("No urgent language detected.")
        
        elif choice == '2':
            if detector.check_suspicious_links():
                print("WARNING: Suspicious links detected - possible phishing attempt.")
            else:
                print("No suspicious links detected.")

        elif choice == '3':
            if detector.analyze_sender_address():
                print("WARNING: Potentially spoofed sender - possible phishing attempt.")
            else:
                print("Sender address appears normal.")
        
        elif choice == '4':
            if detector.check_sensitive_info_requests():
                print("WARNING: Request for sensitive information detected - possible phishing attempt.")
            else:
                print("No requests for sensitive information detected.")
        
        elif choice == '5':
            print("Exiting the Phishing Detector. Stay safe!")
            break
        
        else:
            print("Invalid choice, please enter a number from 1 to 5.")

if __name__ == "__main__":
    main()