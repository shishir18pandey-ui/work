
oracle_db_cbs:
  user: "BOTCBSREAD"

oracle_db_optimus:
  user: "incident_bot_readonly_user"
  password: "ch$9gCjKr5o5mCN#"

# ── DB config (becomes ConfigMap → mounted as /app/db_config.json) ──
db_config:
  cbs:
    oracle:
      main:
        host: "CBSNEWPTDB-scan.idfcbank.com"
        port: 1655
        service_name: "P012SRV"
        tls: true
        ssl_server_cert_dn: "CN=oracletcps_CBSPT.idfcbank.com"
  optimus:
    oracle:
      platform:
        host: "uat-platform-db.ckwkoaqxbuse.ap-south-1.rds.amazonaws.com"
        port: 2484
        service_name: "OPTUATDB"
        tls: true
      payments-core:
        host: "uat-payments-core-db.ckwkoaqxbuse.ap-south-1.rds.amazonaws.com"
        port: 2484
        service_name: "OPTUATDB"
        tls: true
      payments-services:
        host: "uat-payments-services-db.ckwkoaqxbuse.ap-south-1.rds.amazonaws.com"
        port: 2484
        service_name: "OPTUATDB"
        tls: true
      retail-msme:
        host: "uat-retail-msme-db.ckwkoaqxbuse.ap-south-1.rds.amazonaws.com"
        port: 2484
        service_name: "OPTUATDB"
        tls: true
      misc:
        host: "uat-misc-db.ckwkoaqxbuse.ap-south-1.rds.amazonaws.com"
        port: 2484
        service_name: "OPTUATDB"
        tls: true
