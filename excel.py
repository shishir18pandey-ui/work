[
  {
    "name": "idp_customer_username_lookup",
    "type": "SQL",
    "db_instance": "platform",
    "purpose": "Validate username and UCIC mapping in IDP customer table. Used when username mismatch or login issues occur.",
    "query": "SELECT ucic, username FROM IDP.customer WHERE ucic = :ucic",
    "parameters": [
      {"name": "ucic", "type": "string", "description": "Customer UCIC"}
    ],
    "output_description": "Returns UCIC and username mapping.",
    "example_use_cases": ["Incorrect username entered", "Username not mapped to UCIC"],
    "app_tags": ["optimus", "login", "username"]
  },
  {
    "name": "idp_customer_device_lookup",
    "type": "SQL",
    "db_instance": "platform",
    "purpose": "Fetch device_id and user_id for a customer. REQUIRED FIRST STEP before mpin_login_check — MPIN traces in Jaeger are tagged by user_id (UUID), not UCIC, so you must call this to get the user_id. Also used for registration loops, multiple active devices, and MPIN expiry (MPIN_CREATED_ON).",
    "query": "SELECT * FROM IDP.customerdevice WHERE user_id IN (SELECT user_id FROM IDP.customer WHERE ucic = :ucic)",
    "parameters": [
      {"name": "ucic", "type": "string", "description": "Customer UCIC"}
    ],
    "output_description": "Returns device details including DEVICE_ID, USER_ID, MPIN_CREATED_ON, IS_ACTIVE.",
    "example_use_cases": ["Multiple active devices issue", "Registration screen loop", "MPIN expiry validation", "Get user_id before searching MPIN traces"],
    "app_tags": ["optimus", "device", "mpin", "user-id"]
  },
  {
    "name": "idp_get_user_id_from_ucic",
    "type": "SQL",
    "db_instance": "platform",
    "purpose": "MANDATORY FIRST STEP for any mobile-app / MPIN incident. Converts a customer UCIC into the user_id UUID and device_id required by mpin_login_check. Jaeger MPIN traces are tagged ONLY by user_id, so this must be run before any MPIN log search.",
    "query": "SELECT device_id, user_id FROM IDP.customerdevice WHERE user_id IN (SELECT user_id FROM IDP.customer WHERE ucic = :ucic)",
    "parameters": [
      {"name": "ucic", "type": "string", "description": "Customer UCIC"}
    ],
    "output_description": "Returns device_id and user_id (UUID). Pass user_id to mpin_login_check as tag_value with tag_name='user_id'. Multiple rows means multiple registered devices.",
    "example_use_cases": ["Get user_id before searching MPIN traces", "Mobile app login issue", "Forgot MPIN", "Device registration loop"],
    "app_tags": ["optimus", "login", "mpin", "user-id"]
  },
  {
    "name": "idp_device_multiuser_lookup",
    "type": "SQL",
    "db_instance": "platform",
    "purpose": "Check if a device_id is linked to multiple user_ids causing registration/login conflicts.",
    "query": "SELECT * FROM IDP.customerdevice WHERE device_id = :device_id",
    "parameters": [
      {"name": "device_id", "type": "string", "description": "Device identifier"}
    ],
    "output_description": "Returns all user mappings for a device.",
    "example_use_cases": ["Same device mapped to multiple users", "Login conflict due to shared device"],
    "app_tags": ["optimus", "device"]
  },
  {
    "name": "idp_password_expiry_check",
    "type": "SQL",
    "db_instance": "platform",
    "purpose": "Check password status and expiry for login failures.",
    "query": "SELECT ucic, username, passwd_created_on, is_need_to_reset_password FROM IDP.customer WHERE ucic = :ucic",
    "parameters": [
      {"name": "ucic", "type": "string", "description": "Customer UCIC"}
    ],
    "output_description": "Returns password creation date and reset flag.",
    "example_use_cases": ["Password expired", "User locked due to password policy"],
    "app_tags": ["optimus", "login", "password"]
  }
]
