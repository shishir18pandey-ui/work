SERVICE_SELECTION_CONTEXT = {
    "optimus": """
Service names have to be filled as per the following rules:
     idp-api -> Login/authentication/registration issues
     PAYMENTS-API -> Fund transfer problems
     bulk-payments-api -> Bulk transfer problems, bulk_ref_id has to be used here, this has to be referred here
     IPO-API -> Problems related to initial public offerings or IPOs, can only be queried using customer_id
     deposits-api -> Document download, deposit-related issues, FD related issue as well
     CREDIT-CARD-API -> Credit card eligibility, statements, payments, and credit related issue/operations
     ecom-api -> ECOM transaction or ECOM payment and checkout issues, can only be queried using customer_id, txn_id, and txn_request_id
     WEALTH-API -> Issues related to mutual funds such as: View portfolio, Buy/purchase, Redeem mutual funds (Supports tags like customer_id, user_id, or session_tracing_id)
     DEBITCARD-API -> Issues related to debit cards such as: Viewing debit card details, Changing debit card limit, Other debit card related issues (Supports tags like customer_id, user_id)
     upi-api -> Issues related to UPI such as: Transaction, Payment, UPI ID, Link UPI Account
     FX-API -> Issue related to Pay Abroad such as Transaction, payment, link account
     kyc-service -> Issue related to PAN CARD or KYC such as Update, delete, Add Pan Card
     beneficiary-api -> Issues related to adding or verifying beneficiaries or payees, such as: checking if a payee exists, verifying VPA, or failure to add a beneficiary
     EMANDATE -> Issues related to Aadhaar or mandate Create/Delete/Update operations
     CX-API -> Issue related to mobile, mobile number, email, email validation or email related issue
     INSURANCE-API -> Issues related to insurance such as: Unable to view insurance details
     LAS-API -> Issues using LAS-API only (not Loan-API or Wealth-API), covering unclutch/withdraw units, delete sweep, view LAI details, and download LAI statements
     REMITTANCE-API -> Issue related to remittance, invalid remittance or anything related to this keyword
     CAS-API -> Issues related to external mutual fund linking such as: Unable to link external mutual fund, Unable to get OTP while linking external mutual fund, Unable to complete OTP verification while linking external mutual fund, CAS related issues
     E-CHEQUE-API -> Issues related to cheque operations such as getting cheque details or status, placing checkbook requests, stopping or revoking cheques, and updating cheque information
     LOANS-API -> Issues related to loans such as: fetching loan details, downloading loan interest certificates, downloading provisional certificates, downloading welcome letters, downloading repayment schedules
     billpay-api -> Issues related to bill payments such as: Unable to fetch bill details, Unable to do bill payment, Unable to get biller categories, Unable to delete biller, Unable to raise bill disputes, Unable to fetch recharge plans, Bill payment failures, Biller management issues (Supports tags like customer_id, user_id, session_tracing_id, txn_id, mobile_number)
     APPLY-FM-API -> Issues related to new loan applications such as: Unable to apply for loan, Unable to create loan, Unable to create loan lead, NTB (New to Bank) loan application issues, loan application failures (Supports tags like applicationId, ucic, mobile_number, customer_id, user_id)
""",
}


def get_service_selection_context(app: str) -> str:
    if not app:
        return ""
    return SERVICE_SELECTION_CONTEXT.get(app.lower().strip(), "")
