import requests
import json

URL = "https://internal-app.uat-devutils.idfcfirstbank.com/incident_agent/incident/create"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Basic YWRtaW46ZHVtbXktdG9rZW4tMTIzNDU="
}

payload = {
    "message": "Incident has been created successfully.",
    "incidentNumber": "INC000006212999",
    "incidentId": "html_test_001",
    "incidentType": "Application",
    "businessService": "Optimus",
    "tier1": "Optimus",
    "tier2": "Login Issue",
    "tier3": "Testing HTML Encoding",
    "impact": "Low",
    "urgency": "Low",
    "priority": "Low",

    # ── Test case: script tag injection ──
    "shortDescription": "<script>alert('XSS')</script> Customer unable to login",

    "description": "Normal description field, not tested here",
    "contactType": "Self Service",
    "state": "New",
    "assignmentGroup": "Optimus BTO Support",
    "sourceIncidentNum": "INC000006212999",
    "sourceIncidentId": "",

    # ── Test case: URL + special chars + quotes ──
    "businessImpact": "Customer clicked http://malicious-site.com?redirect=<img src=x onerror=alert(1)> & lost access \"urgently\"",

    "cause": "",
    "businessCorrectiveAction": "",
    "techCorrectiveAction": "",
    "dataSource": "",
    "descriptionOfOutage": "",
    "emailID": "",
    "entityUCIC": "1041338998",
    "hashValues": "",
    "ipDetails": "",
    "ldwNotifyInformation": "",
    "loanAccountNumber": "",
    "loginId": "",
    "mobileNumber": "9876567898",
    "businessPreventiveAction": "",
    "techPreventiveAction": "",
    "resoultionTeam": "",
    "rootCause": "",
    "systemName": "",
    "urlOrDomain": "",
    "userDetail": "",
    "individualUCIC": "1041338998",
    "sourceIncCreateddttime": "19-Jun-2026 11:53:42",
    "incidentURL": "https://idfcfirstbanktest2.service-now.com/isupport",
    "userLocation": "",
    "callerId": "sujeet.singh2@idfcfirst.bank.in"
}

response = requests.post(
    URL,
    headers=HEADERS,
    data=json.dumps(payload),
    verify=False
)

print(f"Status: {response.status_code}")
print(f"Response: {response.text}")
