loan_account_patterns = [
    r'\bLoan\s*Account\s*Number[:\s]*(\d{6,20})\b',
    r'\bloan_account_number[:\s]*(\d{6,20})\b',
    r'\bloan\s*account[:\s]*(\d{6,20})\b',
]
for pattern in loan_account_patterns:
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        identifiers["loan_account_number"] = match.group(1)
        break




if "loan_account_number" not in identifiers:
    loan_account_number = payload.get("loanAccountNumber")
    if loan_account_number:
        identifiers["loan_account_number"] = str(loan_account_number)
