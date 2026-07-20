import requests
import json
 
 
COUNT = 1
 
URL = "https://internal-app.uat-devutils.idfcfirstbank.com/incident_agent/incident/create"
 
HEADERS = {
   "Content-Type": "application/json",
   "Authorization": "Basic YWRtaW46ZHVtbXktdG9rZW4tMTIzNDU="
}
 
for i in range(1, COUNT + 1):
 
  payload = {
	"message": "Incident has been created successfully.",
	"incidentNumber": "INC000006212701",
	"incidentId": "1234",
	"incidentType": "Application",
	"businessService": "SFDC Asset Org 2",
	"tier1": "SFDC Asset Org 2",
	"tier2": "Infra",
	"tier3": "Self Monitoring Patrol Agent",
	"impact": "Low",
	"urgency": "Medium",
	"priority": "Low",
	"shortDescription": "27 OCT 2023 Short Description Test",
	"description": "27 OCT 2023 Description Test",
	"contactType": "Event",
	"state": "New",
	"assignmentGroup": "",
	"sourceIncidentNum": "T000123",
	"sourceIncidentId": "26SEP2023-0120-111",
	"businessImpact": "",
	"cause": "",
	"businessCorrectiveAction": "",
	"techCorrectiveAction": "",
	"dataSource": "",
	"descriptionOfOutage": "",
	"emailID": "",
	"entityUCIC": "",
	"hashValues": "",
	"ipDetails": "",
	"ldwNotifyInformation": "",
	"loanAccountNumber": "",
	"loginId": "",
	"mobileNumber": "",
	"businessPreventiveAction": "",
	"techPreventiveAction": "",
	"resoultionTeam": "",
	"rootCause": "",
	"systemName": "",
	"urlOrDomain": "",
	"userDetail": "",
	"individualUCIC": "123",
	"sourceIncCreateddttime": "24-Feb-2023 09:10:04",
	"incidentURL": "https://idfcfirstbanktest2.service-now.com/isupport?sys_idb50370443b90f290b6986f34c3e45a4d&viewsp&idticket&tableincident%22",
	"userLocation": "",
	"files": [
		{
			"fileId": "FILE001",
			"fileName": "pan_card.jpeg",
			"originalFileName": "PAN.jpeg",
			"fileType": "image/jpeg",
			"fileSize": 245678,
			"contentEncoding": "base64",
			"fileContent": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wwAARCAAyAJYDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDEooor0jzQooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigArf0WKKW1ig8mMTzzsiPPAHjm4UeXu6oQT1H94ZIxWBU8N5dW8TxQXM0Ucn31Ryob6gdaTVxp2N+Tw/ZQzQCRrgLi6WZN67g8MQfGduBknBHzYx1pV0HTprq0iia6USXNnHJudT8lwhfj5eqgYz39BWDJqN9Nt829uH2KyrulY4DDawHPQjg+op1rqd1a3cFysrSNDLHKEkYlSY/uZGegHA9BwKmz7lXXY3bDR7O7tZTCrxeevkD7ThmjYXEC7wcDHEhH4MM1l6tY2drFBLZyu4dnRg24gFcdyi+vTBxjrzVR9QvZHLyXlw7MnlktISSuc7evTPamXF3cXbh7m4lmYDAaRyxA9OaaTuJtWOi02wW3S3juoYEnjk1BZDNGHAKW6ld3ByFbJHB74pwks30u4mhl05J0e3jkuJLPMbMRMTtXyzjgJn5RkrXOvfXcjKz3U7MiGNS0hJCkYKj2IJGPei2vryzDi1up4A+NwikK7sZxnHXqfzpcrHzI3jbB/MtYoLN4v7ONzuaLDO3l72ZXC5G1sjGQPlx3rmatDUr4RGIXtwI23Ep5rYO7IbjPfJz65qrTSsJu4UUUVRIUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQB//9k="
		}
   ]
}
 
response = requests.post(
     URL,
     headers=HEADERS,
     data=json.dumps(payload),
     verify=False
     )
 
print(f"Created Incident ID: TEST449001 | Status: {response.status_code}")
