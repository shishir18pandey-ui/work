this is jager_tool.json
[
  {
    "name": "login_password_check",
    "type": "JAEGER",
    "purpose": "Check login/password validation traces to find 'Invalid Credentials' errors. Use for incorrect password, wrong username/UCIC, or mobile entered in username field.",
    "service": "idp-api",
    "parameters": [
      {"name": "tag_name", "type": "string", "description": "Identifier type: ucic, customer_id, mobile_number, username, or user_id", "required": true},
      {"name": "tag_value", "type": "string", "description": "The actual value of the identifier", "required": true}
    ],
    "output_description": "Returns error traces with HTTP status codes and trace links for login failures.",
    "example_use_cases": ["Customer getting Invalid Credentials", "Incorrect password entered", "Wrong username or UCIC", "Password has expired"],
    "app_tags": ["optimus", "login", "authentication"]
  },
  {
    "name": "login_mode_check",
    "type": "JAEGER",
    "purpose": "Check get-login-mode traces to see what USERNAME, CUSTOMER_ID or MOBILE NUMBER the customer entered at login, and validate username/mobile against database. Also validates mobile via CDP endpoint cdp.idfcfirstbank.com/v1/customers/search for primaryMobileWithIsd.",
    "service": "idp-api",
    "parameters": [
      {"name": "tag_name", "type": "string", "description": "Identifier type: ucic, customer_id, mobile_number, username, or user_id", "required": true},
      {"name": "tag_value", "type": "string", "description": "The actual value of the identifier", "required": true}
    ],
    "output_description": "Returns login mode traces showing entered identifier fields and CDP validation results.",
    "example_use_cases": ["Validate entered username", "Validate entered mobile number", "Check login identifier customer entered", "Primary relationship closed using secondary mobile"],
    "app_tags": ["optimus", "login", "identifier"]
  },
  {
    "name": "mpin_login_check",
    "type": "JAEGER",
    "purpose": "Check MPIN login traces to diagnose MPIN expiry or MPIN validation failures. MPIN expires 365 days after creation (MPIN_CREATED_ON in customerdevice table).",
    "service": "idp-api",
    "parameters": [
      {"name": "tag_name", "type": "string", "description": "Identifier type: ucic, customer_id, mobile_number, username, or user_id", "required": true},
      {"name": "tag_value", "type": "string", "description": "The actual value of the identifier", "required": true}
    ],
    "output_description": "Returns MPIN login traces with error codes.",
    "example_use_cases": ["Unable to login via MPIN", "MPIN expired"],
    "app_tags": ["optimus", "login", "mpin"]
  },
  {
    "name": "account_view_check",
    "type": "JAEGER",
    "purpose": "Check account view-details traces to diagnose 'All Good things take time' error when customer cannot view current account. If partnership/sole-prop account, check admin portal setup at https://admin.my.idfcfirstbank.com.",
    "service": "bas-api",
    "parameters": [
      {"name": "tag_name", "type": "string", "description": "Identifier type: ucic, customer_id, mobile_number, or account_number", "required": true},
      {"name": "tag_value", "type": "string", "description": "The actual value of the identifier", "required": true}
    ],
    "output_description": "Returns account view traces. If partnership/sole-prop account, check admin portal setup.",
    "example_use_cases": ["Current account not visible in net/mobile banking", "All Good things take time error"],
    "app_tags": ["optimus", "account", "view"]
  },
  {
    "name": "device_registration_check",
    "type": "JAEGER",
    "purpose": "Check device registration traces for failures including SMS_NOT_SEND (SIMBindingSMSSendFailed), Invalid Credentials during registration, and OTP Verification Failed errors.",
    "service": "idp-api",
    "parameters": [
      {"name": "tag_name", "type": "string", "description": "Identifier type: ucic, customer_id, mobile_number, username, or user_id", "required": true},
      {"name": "tag_value", "type": "string", "description": "The actual value of the identifier", "required": true}
    ],
    "output_description": "Returns registration traces. Look for events: SIMBindingSMSSendFailed/ERR_SMS_NOT_SEND, Credentials are incorrect, OTP Verification Failed.",
    "example_use_cases": ["Unable to register device", "SMS not sent during registration", "Invalid credentials at registration", "OTP verification failed during registration"],
    "app_tags": ["optimus", "registration", "device"]
  },
  {
    "name": "beneficiary_add_check",
    "type": "JAEGER",
    "purpose": "Check beneficiary/payee addition traces to diagnose errors like 'Payee already exists' when customer is unable to add beneficiary details.",
    "service": "beneficiary-api",
    "parameters": [
      {"name": "tag_name", "type": "string", "description": "Identifier type: ucic, customer_id, or mobile_number", "required": true},
      {"name": "tag_value", "type": "string", "description": "The actual value of the identifier", "required": true}
    ],
    "output_description": "Returns beneficiary addition traces with error messages.",
    "example_use_cases": ["Unable to add beneficiary", "Payee already exists error", "Beneficiary addition failed"],
    "app_tags": ["optimus", "beneficiary", "payee"]
  },
  {
    "name": "fund_transfer_check",
    "type": "JAEGER",
    "purpose": "Check fund transfer and UPI transaction traces. Covers payments-api fund transfer endpoint and downstream upi-api merchant/request endpoint. Look for errors like INVALID VIRTUAL ADDRESS from finacq.phi.idfcbank.com.",
    "service": "payments-api",
    "parameters": [
      {"name": "tag_name", "type": "string", "description": "Identifier type: ucic, customer_id, mobile_number, txn_id, or txn_request_id", "required": true},
      {"name": "tag_value", "type": "string", "description": "The actual value of the identifier", "required": true}
    ],
    "output_description": "Returns fund transfer traces. Check downstream upi-api calls and finacq endpoint responses for errors.",
    "example_use_cases": ["Unable to do UPI transaction", "INVALID VIRTUAL ADDRESS error", "Fund transfer failed", "Payment transaction error"],
    "app_tags": ["optimus", "payments", "upi", "fund-transfer"]
  },
  {
    "name": "upi_merchant_request_check",
    "type": "JAEGER",
    "purpose": "Check UPI merchant request/response traces in upi-api service. This is the downstream service called by payments-api for UPI transactions. Look for errors from finacq.phi.idfcbank.com/merchant/request endpoint.",
    "service": "upi-api",
    "parameters": [
      {"name": "tag_name", "type": "string", "description": "Identifier type: ucic, customer_id, mobile_number, txn_id, or txn_request_id", "required": true},
      {"name": "tag_value", "type": "string", "description": "The actual value of the identifier", "required": true}
    ],
    "output_description": "Returns UPI merchant request traces with error details from finacq endpoint.",
    "example_use_cases": ["UPI payment failed", "INVALID VIRTUAL ADDRESS", "Merchant request error"],
    "app_tags": ["optimus", "upi", "merchant"]
  },
  {
    "name": "billpay_transaction_check",
    "type": "JAEGER",
    "purpose": "Check BillPay transaction traces to diagnose failures when customer is unable to pay a bill.",
    "service": "billpay-api",
    "parameters": [
      {"name": "tag_name", "type": "string", "description": "Identifier type: ucic, customer_id, mobile_number, or txn_id", "required": true},
      {"name": "tag_value", "type": "string", "description": "The actual value of the identifier", "required": true}
    ],
    "output_description": "Returns billpay transaction traces with error details.",
    "example_use_cases": ["Unable to do BillPay transaction", "Bill payment failed"],
    "app_tags": ["optimus", "billpay", "payment"]
  },
  {
    "name": "billpay_biller_fetch_check",
    "type": "JAEGER",
    "purpose": "Check biller details fetch traces. This calls a downstream Billdesk endpoint (billdesk-payment-services-sys/validateBDPayment). If no response is found from Billdesk, this is a Billdesk-side issue that needs to be raised with the Billdesk team — not resolvable from our side.",
    "service": "billpay-api",
    "parameters": [
      {"name": "tag_name", "type": "string", "description": "Identifier type: ucic or customer_id", "required": true},
      {"name": "tag_value", "type": "string", "description": "The actual value of the identifier", "required": true}
    ],
    "output_description": "Returns biller fetch traces. If Billdesk endpoint shows no response, this must be escalated to Billdesk team — do not attempt further resolution.",
    "example_use_cases": ["Unable to fetch biller details", "Biller list not loading"],
    "app_tags": ["optimus", "billpay", "biller"]
  },
  {
    "name": "billpay_add_biller_check",
    "type": "JAEGER",
    "purpose": "Check add-new-biller traces. This calls downstream Billdesk endpoint (billdesk-billermgmt-sys/createBDBillerAccount). Common error: customer is trying to add a biller with a short name that already exists in the system for another biller — instruct customer to use a different short name.",
    "service": "billpay-api",
    "parameters": [
      {"name": "tag_name", "type": "string", "description": "Identifier type: ucic or customer_id", "required": true},
      {"name": "tag_value", "type": "string", "description": "The actual value of the identifier", "required": true}
    ],
    "output_description": "Returns add-biller traces. If error indicates duplicate biller name, instruct customer to retry with a different short name.",
    "example_use_cases": ["Unable to add new biller", "Biller name already exists error"],
    "app_tags": ["optimus", "billpay", "biller", "add"]
  },
  {
    "name": "billpay_modify_biller_check",
    "type": "JAEGER",
    "purpose": "Check modify-existing-biller traces. This calls downstream Billdesk endpoint (billdesk-billermgmt-sys/modifyBDBillerAccount). Common error: timeout from Billdesk side — needs escalation to Billdesk team.",
    "service": "billpay-api",
    "parameters": [
      {"name": "tag_name", "type": "string", "description": "Identifier type: ucic or customer_id", "required": true},
      {"name": "tag_value", "type": "string", "description": "The actual value of the identifier", "required": true}
    ],
    "output_description": "Returns modify-biller traces. If timeout from Billdesk, escalate to Billdesk team — not resolvable from our side.",
    "example_use_cases": ["Unable to modify biller details", "Biller update timeout"],
    "app_tags": ["optimus", "billpay", "biller", "modify"]
  },
  {
    "name": "debitcard_view_check",
    "type": "JAEGER",
    "purpose": "Check debit card details view traces. Common cause: customer is in Asset_Only category — debit cards are not issued for asset-only customers.",
    "service": "DEBITCARD-API",
    "parameters": [
      {"name": "tag_name", "type": "string", "description": "Identifier type: ucic or customer_id", "required": true},
      {"name": "tag_value", "type": "string", "description": "The actual value of the identifier", "required": true}
    ],
    "output_description": "Returns debit card view traces. Check if customer falls under Asset_Only category — if so, this is expected behavior, not a bug.",
    "example_use_cases": ["Unable to view debit card details", "Debit card not showing"],
    "app_tags": ["optimus", "debitcard", "view"]
  },
  {
    "name": "debitcard_international_limit_check",
    "type": "JAEGER",
    "purpose": "Check traces for setting international transaction limits on debit card. This calls downstream SFDC endpoint (generic-case-exp/cardServices) present in the same trace.",
    "service": "DEBITCARD-API",
    "parameters": [
      {"name": "tag_name", "type": "string", "description": "Identifier type: ucic or customer_id", "required": true},
      {"name": "tag_value", "type": "string", "description": "The actual value of the identifier", "required": true}
    ],
    "output_description": "Returns international limit traces including downstream SFDC errors.",
    "example_use_cases": ["Unable to set international debit card limits", "Card limit update failed"],
    "app_tags": ["optimus", "debitcard", "limit"]
  },
  {
    "name": "debitcard_virtual_card_check",
    "type": "JAEGER",
    "purpose": "Check virtual debit card request traces. This calls downstream DCMS endpoint (card-mgmt-proc/createVirtualCard). Common error: timeout from DCMS side.",
    "service": "DEBITCARD-API",
    "parameters": [
      {"name": "tag_name", "type": "string", "description": "Identifier type: ucic or customer_id", "required": true},
      {"name": "tag_value", "type": "string", "description": "The actual value of the identifier", "required": true}
    ],
    "output_description": "Returns virtual card request traces. If DCMS timeout, this is a downstream DCMS issue.",
    "example_use_cases": ["Unable to request virtual debit card", "Virtual card creation timeout"],
    "app_tags": ["optimus", "debitcard", "virtual"]
  },
  {
    "name": "debitcard_pin_generation_check",
    "type": "JAEGER",
    "purpose": "Check debit card PIN generation traces. This calls downstream DCMS endpoint (fss-card-services-sys/generatePin). The actual error message is encrypted in the response and needs decryption to identify root cause.",
    "service": "DEBITCARD-API",
    "parameters": [
      {"name": "tag_name", "type": "string", "description": "Identifier type: ucic or customer_id", "required": true},
      {"name": "tag_value", "type": "string", "description": "The actual value of the identifier", "required": true}
    ],
    "output_description": "Returns PIN generation traces. Note: actual DCMS error message may be encrypted in response payload.",
    "example_use_cases": ["Unable to generate debit card PIN", "PIN generation failed"],
    "app_tags": ["optimus", "debitcard", "pin"]
  },
  {
    "name": "fixed_deposit_view_check",
    "type": "JAEGER",
    "purpose": "Check fixed deposit details view traces. Common errors: (1) 'unauthorized to perform action' — customer does not have deposit permissions, (2) timeout from CBS endpoint (bancs/fdList) — CBS-side issue.",
    "service": "deposit-api",
    "parameters": [
      {"name": "tag_name", "type": "string", "description": "Identifier type: ucic or customer_id", "required": true},
      {"name": "tag_value", "type": "string", "description": "The actual value of the identifier", "required": true}
    ],
    "output_description": "Returns FD view traces. If 'unauthorized' error, customer lacks deposit permissions — not a bug. If CBS timeout, escalate to CBS team.",
    "example_use_cases": ["Unable to view fixed deposit details", "FD details not loading", "FD business account view issue"],
    "app_tags": ["optimus", "deposit", "fixed-deposit", "view"]
  },
  {
    "name": "fixed_deposit_advice_download_check",
    "type": "JAEGER",
    "purpose": "Check FD advice download traces. Common error: 'unauthorized to perform action' — customer does not have deposit permissions.",
    "service": "deposits-api",
    "parameters": [
      {"name": "tag_name", "type": "string", "description": "Identifier type: ucic or customer_id", "required": true},
      {"name": "tag_value", "type": "string", "description": "The actual value of the identifier", "required": true}
    ],
    "output_description": "Returns FD advice download traces. If unauthorized error, customer lacks deposit permissions.",
    "example_use_cases": ["Unable to download FD advice"],
    "app_tags": ["optimus", "deposit", "advice"]
  },
  {
    "name": "fixed_deposit_tds_check",
    "type": "JAEGER",
    "purpose": "Check TDS certificate download traces for fixed deposits. Common error: 'unauthorized to perform action' — customer does not have deposit permissions.",
    "service": "deposits-api",
    "parameters": [
      {"name": "tag_name", "type": "string", "description": "Identifier type: ucic or customer_id", "required": true},
      {"name": "tag_value", "type": "string", "description": "The actual value of the identifier", "required": true}
    ],
    "output_description": "Returns TDS download traces. If unauthorized error, customer lacks deposit permissions.",
    "example_use_cases": ["Unable to download TDS certificate"],
    "app_tags": ["optimus", "deposit", "tds"]
  },
  {
    "name": "form121_eligibility_check",
    "type": "JAEGER",
    "purpose": "Check Form 121 eligibility traces. Common error: 'Country of tax residence not eligible'.",
    "service": "deposits-api",
    "parameters": [
      {"name": "tag_name", "type": "string", "description": "Identifier type: ucic or customer_id", "required": true},
      {"name": "tag_value", "type": "string", "description": "The actual value of the identifier", "required": true}
    ],
    "output_description": "Returns Form 121 eligibility traces. If country of tax residence not eligible, this is expected — customer's country doesn't qualify.",
    "example_use_cases": ["Unable to submit Form 121", "Form 121 eligibility failed"],
    "app_tags": ["optimus", "deposit", "form121", "tax"]
  },
  {
    "name": "credit_card_view_check",
    "type": "JAEGER",
    "purpose": "Check credit card details view traces. Calls downstream Prime endpoint (ccei/creditcard/entityinquiry). Common error: timeout from Prime end.",
    "service": "CREDIT-CARD-API",
    "parameters": [
      {"name": "tag_name", "type": "string", "description": "Identifier type: ucic or customer_id", "required": true},
      {"name": "tag_value", "type": "string", "description": "The actual value of the identifier", "required": true}
    ],
    "output_description": "Returns credit card view traces. If Prime timeout, this is a downstream Prime-side issue.",
    "example_use_cases": ["Unable to view credit card details", "Credit card not loading"],
    "app_tags": ["optimus", "creditcard", "view"]
  },
  {
    "name": "credit_card_balance_transfer_check",
    "type": "JAEGER",
    "purpose": "Check balance transfer initiation traces. Common failure: 'no active offer available'. Verify downstream offer service call to customer-generic-info-exp/v2/offer.",
    "service": "CREDIT-CARD-API",
    "parameters": [
      {
        "name": "tag_name",
        "type": "string",
        "description": "Identifier type: customer_id or ucic",
        "required": true
      },
      {
        "name": "tag_value",
        "type": "string",
        "description": "The actual identifier value",
        "required": true
      }
    ],
    "output_description": "Returns balance transfer traces and downstream offer API response. If 'no active offer available' is returned, customer is currently not eligible for a balance transfer offer.",
    "example_use_cases": [
      "Unable to transfer credit card balance",
      "No active offer available",
      "Balance transfer initiation failed"
    ],
    "app_tags": ["optimus", "creditcard", "balance-transfer"]
  },
  {
    "name": "credit_card_addon_application_check",
    "type": "JAEGER",
    "purpose": "Check Add-on Credit Card application traces. Verify downstream call to creditcard-application-exp/v1/createAddOnAppln for application errors.",
    "service": "CREDIT-CARD-API",
    "parameters": [
      {
        "name": "tag_name",
        "type": "string",
        "description": "Identifier type: customer_id or ucic",
        "required": true
      },
      {
        "name": "tag_value",
        "type": "string",
        "description": "The actual identifier value",
        "required": true
      }
    ],
    "output_description": "Returns Add-on Credit Card application traces and downstream application service response.",
    "example_use_cases": [
      "Unable to apply for Add-on Credit Card",
      "Add-on card application failed"
    ],
    "app_tags": ["optimus", "creditcard", "addon-card"]
  },
  {
    "name": "payments_add_funds_check",
    "type": "JAEGER",
    "purpose": "Check Add Funds transaction traces. Common error: 'account not found' while adding funds.",
    "service": "payments-api",
    "parameters": [
      {
        "name": "tag_name",
        "type": "string",
        "description": "Identifier type: customer_id, account_number or txn_id",
        "required": true
      },
      {
        "name": "tag_value",
        "type": "string",
        "description": "The actual identifier value",
        "required": true
      }
    ],
    "output_description": "Returns Add Funds transaction traces including account validation errors.",
    "example_use_cases": [
      "Unable to add funds",
      "Account not found",
      "Add funds transaction failed"
    ],
    "app_tags": ["optimus", "payments", "add-funds"]
  },
  {
    "name": "mutual_fund_holdings_check",
    "type": "JAEGER",
    "purpose": "Check Mutual Fund holdings traces when customer is unable to view investment details.",
    "service": "WEALTH-API",
    "parameters": [
      {
        "name": "tag_name",
        "type": "string",
        "description": "Identifier type: customer_id or ucic",
        "required": true
      },
      {
        "name": "tag_value",
        "type": "string",
        "description": "The actual identifier value",
        "required": true
      }
    ],
    "output_description": "Returns Mutual Fund holdings traces and downstream errors, if any.",
    "example_use_cases": [
      "Unable to view mutual fund holdings",
      "Investment details not loading",
      "Mutual fund portfolio unavailable"
    ],
    "app_tags": ["optimus", "wealth", "mutual-fund", "investment"]
  }
]
this si sql_tool.json 
[
  {
    "name": "idp_customer_username_lookup",
    "type": "SQL",
    "db_instance": "platform",
    "purpose": "Validate username and UCIC mapping in IDP customer table. Used when username mismatch or login issues occur.",
    "query": "SELECT ucic, username FROM IDP.customer WHERE ucic = :ucic",
    "parameters": [
      {
        "name": "ucic",
        "type": "string",
        "description": "Customer UCIC"
      }
    ],
    "output_description": "Returns UCIC and username mapping.",
    "example_use_cases": [
      "Incorrect username entered",
      "Username not mapped to UCIC"
    ],
    "app_tags": ["optimus", "login", "username"]
  },
  {
    "name": "idp_customer_device_lookup",
    "type": "SQL",
    "db_instance": "platform",
    "purpose": "Fetch all devices linked to a customer user_id. Used for registration loops, device mismatch, and MPIN issues.",
    "query": "SELECT * FROM IDP.customerdevice WHERE user_id IN (SELECT user_id FROM IDP.customer WHERE ucic = :ucic)",
    "parameters": [
      {
        "name": "ucic",
        "type": "string",
        "description": "Customer UCIC"
      }
    ],
    "output_description": "Returns device details including DEVICE_ID, MPIN_CREATED_ON, IS_ACTIVE.",
    "example_use_cases": [
      "Multiple active devices issue",
      "Registration screen loop",
      "MPIN expiry validation"
    ],
    "app_tags": ["optimus", "device", "mpin"]
  },
  {
    "name": "idp_device_multiuser_lookup",
    "type": "SQL",
    "db_instance": "platform",
    "purpose": "Check if a device_id is linked to multiple user_ids causing registration/login conflicts.",
    "query": "SELECT * FROM IDP.customerdevice WHERE device_id = :device_id",
    "parameters": [
      {
        "name": "device_id",
        "type": "string",
        "description": "Device identifier"
      }
    ],
    "output_description": "Returns all user mappings for a device.",
    "example_use_cases": [
      "Same device mapped to multiple users",
      "Login conflict due to shared device"
    ],
    "app_tags": ["optimus", "device"]
  },
  {
    "name": "idp_password_expiry_check",
    "type": "SQL",
    "db_instance": "platform",
    "purpose": "Check password status and expiry for login failures.",
    "query": "SELECT ucic, username, passwd_created_on, is_need_to_reset_password FROM IDP.customer WHERE ucic = :ucic",
    "parameters": [
      {
        "name": "ucic",
        "type": "string",
        "description": "Customer UCIC"
      }
    ],
    "output_description": "Returns password creation date and reset flag.",
    "example_use_cases": [
      "Password expired",
      "User locked due to password policy"
    ],
    "app_tags": ["optimus", "login", "password"]
  }
]



now see these are new excel sheet , can u cretae the new jager tool and sql tool base don this :
