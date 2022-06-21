"use strict";(self.webpackChunkjackmleitch_com_np=self.webpackChunkjackmleitch_com_np||[]).push([[2839],{3905:function(t,e,a){a.d(e,{Zo:function(){return d},kt:function(){return h}});var n=a(7294);function i(t,e,a){return e in t?Object.defineProperty(t,e,{value:a,enumerable:!0,configurable:!0,writable:!0}):t[e]=a,t}function r(t,e){var a=Object.keys(t);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(t);e&&(n=n.filter((function(e){return Object.getOwnPropertyDescriptor(t,e).enumerable}))),a.push.apply(a,n)}return a}function s(t){for(var e=1;e<arguments.length;e++){var a=null!=arguments[e]?arguments[e]:{};e%2?r(Object(a),!0).forEach((function(e){i(t,e,a[e])})):Object.getOwnPropertyDescriptors?Object.defineProperties(t,Object.getOwnPropertyDescriptors(a)):r(Object(a)).forEach((function(e){Object.defineProperty(t,e,Object.getOwnPropertyDescriptor(a,e))}))}return t}function o(t,e){if(null==t)return{};var a,n,i=function(t,e){if(null==t)return{};var a,n,i={},r=Object.keys(t);for(n=0;n<r.length;n++)a=r[n],e.indexOf(a)>=0||(i[a]=t[a]);return i}(t,e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(t);for(n=0;n<r.length;n++)a=r[n],e.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(t,a)&&(i[a]=t[a])}return i}var l=n.createContext({}),c=function(t){var e=n.useContext(l),a=e;return t&&(a="function"==typeof t?t(e):s(s({},e),t)),a},d=function(t){var e=c(t.components);return n.createElement(l.Provider,{value:e},t.children)},u={inlineCode:"code",wrapper:function(t){var e=t.children;return n.createElement(n.Fragment,{},e)}},p=n.forwardRef((function(t,e){var a=t.components,i=t.mdxType,r=t.originalType,l=t.parentName,d=o(t,["components","mdxType","originalType","parentName"]),p=c(a),h=i,m=p["".concat(l,".").concat(h)]||p[h]||u[h]||r;return a?n.createElement(m,s(s({ref:e},d),{},{components:a})):n.createElement(m,s({ref:e},d))}));function h(t,e){var a=arguments,i=e&&e.mdxType;if("string"==typeof t||i){var r=a.length,s=new Array(r);s[0]=p;var o={};for(var l in e)hasOwnProperty.call(e,l)&&(o[l]=e[l]);o.originalType=t,o.mdxType="string"==typeof t?t:i,s[1]=o;for(var c=2;c<r;c++)s[c]=a[c];return n.createElement.apply(null,s)}return n.createElement.apply(null,a)}p.displayName="MDXCreateElement"},8770:function(t,e,a){a.r(e),a.d(e,{assets:function(){return d},contentTitle:function(){return l},default:function(){return h},frontMatter:function(){return o},metadata:function(){return c},toc:function(){return u}});var n=a(7462),i=a(3366),r=(a(7294),a(3905)),s=["components"],o={slug:"Strava-Data-Pipeline",title:"Building a ELT Strava Data Pipline",tags:["Python","AWS","Airflow","Data-Engineering"],authors:"jack"},l=void 0,c={permalink:"/blog/Strava-Data-Pipeline",source:"@site/blog/2022-06-20-StravaDataPipeline.md",title:"Building a ELT Strava Data Pipline",description:"EtLT of my own Strava data using the Strava API, MySQL, Python, S3, Redshift, and Airflow",date:"2022-06-20T00:00:00.000Z",formattedDate:"June 20, 2022",tags:[{label:"Python",permalink:"/blog/tags/python"},{label:"AWS",permalink:"/blog/tags/aws"},{label:"Airflow",permalink:"/blog/tags/airflow"},{label:"Data-Engineering",permalink:"/blog/tags/data-engineering"}],readingTime:10.27,truncated:!0,authors:[{name:"Jack Leitch",title:"Machine Learning Engineer",url:"https://github.com/jackmleitch",imageURL:"https://github.com/jackmleitch.png",key:"jack"}],frontMatter:{slug:"Strava-Data-Pipeline",title:"Building a ELT Strava Data Pipline",tags:["Python","AWS","Airflow","Data-Engineering"],authors:"jack"},nextItem:{title:"Building NLP Powered Applications with Hugging Face Transformers",permalink:"/blog/Essay-Companion"}},d={authorsImageUrls:[void 0]},u=[{value:"Data Extraction",id:"data-extraction",level:2},{value:"Data Loading",id:"data-loading",level:2},{value:"Data Validation",id:"data-validation",level:2},{value:"Data Transformations",id:"data-transformations",level:2},{value:"Putting it All Together with Airflow",id:"putting-it-all-together-with-airflow",level:2},{value:"Data Visualization",id:"data-visualization",level:2},{value:"Unit Testing",id:"unit-testing",level:2}],p={toc:u};function h(t){var e=t.components,o=(0,i.Z)(t,s);return(0,r.kt)("wrapper",(0,n.Z)({},p,o,{components:e,mdxType:"MDXLayout"}),(0,r.kt)("p",null,(0,r.kt)("strong",{parentName:"p"},"EtLT of my own Strava data using the Strava API, MySQL, Python, S3, Redshift, and Airflow")),(0,r.kt)("p",null,(0,r.kt)("img",{alt:"system_diagram",src:a(6527).Z,width:"1450",height:"718"})),(0,r.kt)("p",null,(0,r.kt)("strong",{parentName:"p"},"I build an EtLT pipeline to ingest my ",(0,r.kt)("a",{parentName:"strong",href:"https://www.strava.com/athletes/5028644"},"Strava data")," from the Strava API and load it into a ",(0,r.kt)("a",{parentName:"strong",href:"https://aws.amazon.com/redshift/"},"Redshift")," data warehouse. This pipeline is then run once a week using ",(0,r.kt)("a",{parentName:"strong",href:"https://airflow.apache.org"},"Airflow")," to extract any new activity data. The end goal is then to use this data warehouse to build an automatically updating dashboard in Tableau and also to trigger automatic re-training of my ",(0,r.kt)("a",{parentName:"strong",href:"https://github.com/jackmleitch/StravaKudos"},"Strava Kudos Prediction model"),".")),(0,r.kt)("h2",{id:"data-extraction"},(0,r.kt)("a",{parentName:"h2",href:"https://github.com/jackmleitch/StravaDataPipline/blob/master/src/extract_strava_data.py"},"Data Extraction")),(0,r.kt)("p",null,"My Strava activity data is first ",(0,r.kt)("strong",{parentName:"p"},"ingested incrementally")," using the ",(0,r.kt)("a",{parentName:"p",href:"https://developers.strava.com"},"Strava API")," and\nloaded into an ",(0,r.kt)("strong",{parentName:"p"},"S3 bucket"),". On each ingestion run, we query a MySQL database to get the date of the last extraction:"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},'def get_date_of_last_warehouse_update() -> Tuple[datetime, str]:\n    """\n    Get the datetime of last time data was extracted from Strava API\n    by querying MySQL database and also return current datetime.\n    """\n    mysql_conn = connect_mysql()\n    get_last_updated_query = """\n        SELECT COALESCE(MAX(LastUpdated), \'1900-01-01\')\n        FROM last_extracted;"""\n    mysql_cursor = mysql_conn.cursor()\n    mysql_cursor.execute(get_last_updated_query)\n    result = mysql_cursor.fetchone()\n    last_updated_warehouse = datetime.strptime(result[0], "%Y-%m-%d %H:%M:%S")\n    current_datetime = datetime.today().strftime("%Y-%m-%d %H:%M:%S")\n    return last_updated_warehouse, current_datetime\n')),(0,r.kt)("p",null,"We then make repeated calls to the REST API using the ",(0,r.kt)("inlineCode",{parentName:"p"},"requests")," library until we have all activity data between now and ",(0,r.kt)("inlineCode",{parentName:"p"},"last_updated_warehouse"),". We include a ",(0,r.kt)("inlineCode",{parentName:"p"},"time.sleep()")," command to comply with Strava's set rate limit of 100 requests/15 minutes. We also include ",(0,r.kt)("inlineCode",{parentName:"p"},"try: except:")," blocks to combat\nmissing data on certain activities."),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},'def make_strava_api_request(\n    header: Dict[str, str], activity_num: int = 1\n) -> Dict[str, str]:\n    """Use Strava API to get recent page of new data."""\n    param = {"per_page": 1, "page": activity_num}\n    api_response = requests.get(\n        "https://www.strava.com/api/v3/athlete/activities", headers=header, params=param\n    ).json()\n    response_json = api_response[0]\n    return response_json\n\ndef extract_strava_activities(last_updated_warehouse: datetime) -> List[List]:\n    """Connect to Strava API and get data up until last_updated_warehouse datetime."""\n    header = connect_strava()\n    all_activities = []\n    activity_num = 1\n    # while activity has not been extracted yet\n    while True:\n        # Strava has a rate limit of 100 requests every 15 mins\n        if activity_num % 75 == 0:\n            print("Rate limit hit, sleeping for 15 minutes...")\n            time.sleep(15 * 60)\n        try:\n            response_json = make_strava_api_request(header, activity_num)\n        # rate limit has exceeded, wait 15 minutes\n        except KeyError:\n            print("Rate limit hit, sleeping for 15 minutes...")\n            time.sleep(15 * 60)\n            response_json = make_strava_api_request(header, activity_num)\n        date = response_json["start_date"]\n        if date > last_updated_warehouse:\n            activity = parse_api_output(response_json)\n            all_activities.append(activity)\n            activity_num += 1\n        else:\n            break\n    return all_activities\n')),(0,r.kt)("p",null,"Before exporting the data locally into a flat pipe-delimited ",(0,r.kt)("inlineCode",{parentName:"p"},".csv")," file, we perform a few minor transformations such as formatting dates and timezone columns. Hence the little 't' in EtLT! After we save the data, it is then uploaded to an S3 bucket for later loading into the data warehouse."),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},'def save_data_to_csv(all_activities: List[List]) -> str:\n    """Save extracted data to .csv file."""\n    todays_date = datetime.today().strftime("%Y_%m_%d")\n    export_file_path = f"strava_data/{todays_date}_export_file.csv"\n    with open(export_file_path, "w") as fp:\n        csvw = csv.writer(fp, delimiter="|")\n        csvw.writerows(all_activities)\n    return export_file_path\n\ndef upload_csv_to_s3(export_file_path: str) -> None:\n    """Upload extracted .csv file to s3 bucket."""\n    s3 = connect_s3()\n    s3.upload_file(export_file_path, "strava-data-pipeline", export_file_path)\n')),(0,r.kt)("p",null,"Finally, we execute a query to update the MySQL database on the last date of extraction."),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},'def save_extraction_date_to_database(current_datetime: datetime) -> None:\n    """Update last extraction date in MySQL database to todays datetime."""\n    mysql_conn = connect_mysql()\n    update_last_updated_query = """\n        INSERT INTO last_extracted (LastUpdated)\n        VALUES (%s);"""\n    mysql_cursor = mysql_conn.cursor()\n    mysql_cursor.execute(update_last_updated_query, current_datetime)\n    mysql_conn.commit()\n')),(0,r.kt)("h2",{id:"data-loading"},(0,r.kt)("a",{parentName:"h2",href:"https://github.com/jackmleitch/StravaDataPipline/blob/master/src/copy_to_redshift_staging.py"},"Data Loading")),(0,r.kt)("p",null,"Once the data is loaded into the S3 data lake it is then loaded into our ",(0,r.kt)("strong",{parentName:"p"},"Redshift")," data warehouse. We load the data in two parts:"),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},"We first load the data from the S3 bucket into a staging table with the same schema as our production table"),(0,r.kt)("li",{parentName:"ul"},"We then perform validation tests between the staging table and the production table (see ",(0,r.kt)("a",{parentName:"li",href:"#data-validation"},"here"),"). If all critical tests pass we then remove all duplicates between the two tables by first deleting them from the production table. The data from the staging table is then fully inserted into the production table.")),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},'def copy_to_redshift_staging(table_name: str, rs_conn, s3_file_path: str, role_string: str) -> None:\n    """Copy data from s3 into Redshift staging table."""\n    # write queries to execute on redshift\n    create_temp_table = f"CREATE TABLE staging_table (LIKE {table_name});"\n    sql_copy_to_temp = f"COPY staging_table FROM {s3_file_path} iam_role {role_string};"\n\n    # execute queries\n    cur = rs_conn.cursor()\n    cur.execute(create_temp_table)\n    cur.execute(sql_copy_to_temp)\n    rs_conn.commit()\n\ndef redshift_staging_to_production(table_name: str, rs_conn) -> None:\n    """Copy data from Redshift staging table to production table."""\n    # if id already exists in table, we remove it and add new id record during load\n    delete_from_table = f"DELETE FROM {table_name} USING staging_table WHERE \'{table_name}\'.id = staging_table.id;"\n    insert_into_table = f"INSERT INTO {table_name} SELECT * FROM staging_table;"\n    drop_temp_table = "DROP TABLE staging_table;"\n    # execute queries\n    cur = rs_conn.cursor()\n    cur.execute(delete_from_table)\n    cur.execute(insert_into_table)\n    cur.execute(drop_temp_table)\n    rs_conn.commit()\n')),(0,r.kt)("h2",{id:"data-validation"},(0,r.kt)("a",{parentName:"h2",href:"https://github.com/jackmleitch/StravaDataPipline/blob/master/src/validator.py"},"Data Validation")),(0,r.kt)("p",null,"We implement a simple framework in python that is used to execute SQL-based data validation checks in our data pipeline. Although it lacks many features we would expect to see in a production environment, it is a good start and provides some insight into how we can improve our infrastructure."),(0,r.kt)("p",null,"The ",(0,r.kt)("inlineCode",{parentName:"p"},"validatior.py")," script executes a pair of SQL scripts on Redshift and compares the two based on a comparison operator (>, <, =). The test then passes or fails based on the outcome of the two executed scripts. We execute this validation step after we upload our newly ingested data to the staging table but before we insert this table into the production table."),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},'def execute_test(db_conn, script_1: str, script_2: str, comp_operator: str) -> bool:\n    """\n    Execute test made up of two scripts and a comparison operator\n    :param comp_operator: comparison operator to compare script outcome\n        (equals, greater_equals, greater, less_equals, less, not_equals)\n    :return: True/False for test pass/fail\n    """\n    # execute the 1st script and store the result\n    cursor = db_conn.cursor()\n    sql_file = open(script_1, "r")\n    cursor.execute(sql_file.read())\n    record = cursor.fetchone()\n    result_1 = record[0]\n    db_conn.commit()\n    cursor.close()\n\n    # execute the 2nd script and store the result\n    ...\n\n    print("Result 1 = " + str(result_1))\n    print("Result 2 = " + str(result_2))\n\n    # compare values based on the comp_operator\n    if comp_operator == "equals": return result_1 == result_2\n    elif comp_operator == "greater_equals": return result_1 >= result_2\n    ...\n\n    # tests have failed if we make it here\n    return False\n')),(0,r.kt)("p",null,"As a starting point, I implemented checks that check for duplicates, compare the distribution of the total activities in the staging table (Airflow is set to execute at the end of each week) to the average historical weekly activity count, and compare the distribution of the Kudos Count metric to the historical distribution using the z-score. In other words, the last two queries check if the values are within a 90% confidence interval in either direction of what's expected based on history. For example, the following query computes the z-score for the total activities uploaded in a given week (found in the staging table)."),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-sql"},"with activities_by_week AS (\n  SELECT\n    date_trunc('week', start_date::date) AS activity_week,\n    COUNT(*) AS activity_count\n  FROM public.strava_activity_data\n  GROUP BY activity_week\n  ORDER BY activity_week\n),\n\nactivities_by_week_statistics AS (\n  SELECT\n    AVG(activity_count) AS avg_activities_per_week,\n    STDDEV(activity_count) AS std_activities_per_week\n  FROM activities_by_week\n),\n\nstaging_table_weekly_count AS (\n  SELECT COUNT(*) AS staging_weekly_count\n  FROM staging_table\n),\n\nactivity_count_zscore AS (\n  SELECT\n    s.staging_weekly_count AS staging_table_count,\n    p.avg_activities_per_week AS avg_activities_per_week,\n    p.std_activities_per_week as std_activities_per_week,\n    --compute zscore for weekly activity count\n    (staging_table_count - avg_activities_per_week) / std_activities_per_week AS z_score\n  FROM staging_table_weekly_count s, activities_by_week_statistics p\n)\n\nSELECT ABS(z_score) AS two_sized_zscore\nFROM activity_count_zscore;\n")),(0,r.kt)("p",null,"By running"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-sh"},"python src/validator.py sql/validation/weekly_activity_count_zscore.sql sql/validation/zscore_90_twosided.sql.sql greater_equals warn`\n")),(0,r.kt)("p",null,"in the terminal we compare this z-score found in the previous query to the 90% confidence interval z-score ",(0,r.kt)("inlineCode",{parentName:"p"},"SELECT 1.645;"),". The 'warn' at the end of the command tells the script not to exit with an error but to warn us instead. On the other hand, if we add 'halt' to the end the script will exit with an error code and halt all further downstream tasks."),(0,r.kt)("p",null,"We also implement a system to send a notification to a given Slack channel with the validation test results, this validation system was inspired by the Data Validation in Pipelines chapter of James Densmore's excellent Data Pipelines book."),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},'def send_slack_notification(webhook_url: str, script_1: str, script_2: str,\n    comp_operator: str, test_result: bool) -> bool:\n    try:\n        if test_result == True:\n            message = (f"Validation Test Passed!: {script_1} / {script_2} / {comp_operator}")\n        else:\n            message = (f"Validation Test FAILED!: {script_1} / {script_2} / {comp_operator}")\n        # send test result to Slack\n        slack_data = {"text": message}\n        response = requests.post(webhook_url, data=json.dumps(slack_data),\n            headers={"Content-Type": "application/json"})\n        # if post request to Slack fails\n        if response.status_code != 200:\n            print(response)\n            return False\n    except Exception as e:\n        print("Error sending slack notification")\n        print(str(e))\n        return False\n')),(0,r.kt)("p",null,"We then combine all the tests to a shell script ",(0,r.kt)("inlineCode",{parentName:"p"},"validate_load_data.sh")," that we run after loading the data from the S3 bucket to a staging table but before we insert this data into the production table. Running this pipeline on last week's data gives us the following output:\n",(0,r.kt)("img",{alt:"slack",src:a(6975).Z,width:"1974",height:"238"}),"\nIt's great to see that our second test failed because I didn't run anywhere near as much last week as I usually do!"),(0,r.kt)("p",null,"Although this validation framework is very basic, it is a good foundation that can be built upon at a later date."),(0,r.kt)("h2",{id:"data-transformations"},(0,r.kt)("a",{parentName:"h2",href:"https://github.com/jackmleitch/StravaDataPipline/blob/master/src/build_data_model.py"},"Data Transformations")),(0,r.kt)("p",null,"Now the data has been ingested into the data warehouse, the next step in the pipeline is data transformations. Data transformations in this case include both non-contextual manipulation of the data and modeling of the data with context and logic in mind. The benefit of using the ELT methodology instead of the ETL framework, in this case, is that it gives us, the end-user, the freedom to transform the data the way we need as opposed to having a fixed data model that we cannot change (or at least not change without hassle). In my case, I am connecting my Redshift data warehouse to Tableau building out a dashboard. We can, for example, build a data model to extract monthly statistics:"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-sql"},"CREATE TABLE IF NOT EXISTS activity_summary_monthly (\n  activity_month numeric,\n  ...\n  std_kudos real\n);\n\nTRUNCATE activity_summary_monthly;\n\nINSERT INTO activity_summary_monthly\nSELECT EXTRACT(MONTH FROM start_date) AS activity_month,\n    ROUND(SUM(distance)/1609) AS total_miles_ran,\n    ...\n    ROUND(STDDEV(kudos_count), 1) AS std_kudos\nFROM public.strava_activity_data\nWHERE type='Run'\nGROUP BY activity_month\nORDER BY activity_month;\n")),(0,r.kt)("p",null,"We can also build more complicated data models. For example, we can get the week-by-week percentage change in total weekly kudos broken down by workout type:"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-sql"},"WITH weekly_kudos_count AS (\n  SELECT DATE_PART('week', start_date) AS week_of_year,\n    workout_type,\n    SUM(kudos_count) AS total_kudos\n  FROM public.strava_activity_data\n  WHERE type = 'Run' AND DATE_PART('year', start_date) = '2022'\n  GROUP BY week_of_year, workout_type\n),\n\nweekly_kudos_count_lag AS (\n  SELECT *,\n    LAG(total_kudos) OVER(PARTITION BY workout_type ORDER BY week_of_year)\n        AS previous_week_total_kudos\n  FROM weekly_kudos_count\n)\n\nSELECT *,\n    COALESCE(ROUND(((total_kudos - previous_week_total_kudos)/previous_week_total_kudos)*100),0)\n        AS percent_kudos_change\nFROM weekly_kudos_count_lag;\n")),(0,r.kt)("p",null,"A further direction to take this would be to utilize a 3rd party tool such as ",(0,r.kt)("a",{parentName:"p",href:"https://www.getdbt.com"},"dbt")," to implement data modeling."),(0,r.kt)("h2",{id:"putting-it-all-together-with-airflow"},(0,r.kt)("a",{parentName:"h2",href:"https://github.com/jackmleitch/StravaDataPipline/blob/master/airflow/dags/elt_strava_pipeline.py"},"Putting it All Together with Airflow")),(0,r.kt)("p",null,"We create a DAG to orchestrate our data pipeline. We set the pipeline to run weekly which means it will run once a week at midnight on Sunday morning. As seen in the diagram below, our DAG will:"),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},"First, extract any recent data using the Strava API and upload it to an S3 bucket"),(0,r.kt)("li",{parentName:"ul"},"It will then load this data into a staging table in our Redshift cluster"),(0,r.kt)("li",{parentName:"ul"},"The 3 validation tests will then be executed, messaging our Slack channel the results"),(0,r.kt)("li",{parentName:"ul"},"The staging table will then be inserted into the production table, removing any duplicates in the process"),(0,r.kt)("li",{parentName:"ul"},"Finally, a monthly aggregation data model will be created in a new table ",(0,r.kt)("inlineCode",{parentName:"li"},"activity_summary_monthly"))),(0,r.kt)("p",null,(0,r.kt)("img",{alt:"dag",src:a(1229).Z,width:"2360",height:"552"})),(0,r.kt)("h2",{id:"data-visualization"},"Data Visualization"),(0,r.kt)("p",null,"With the data transformations done we were then able to build out an interactive dashboard using Tableau that updates automatically when new data gets ingested to the data warehouse, which is weekly. The dashboard I created was built to investigate how Kudos on my Strava activities changes over time and location. After building this project I shut down the Redshift server to not incur any costs but a screenshot of the dashboard can be seen below.\n",(0,r.kt)("img",{alt:"dashboard",src:a(2682).Z,width:"2480",height:"1008"}),"\n",(0,r.kt)("img",{alt:"dashboard",src:a(1336).Z,width:"1461",height:"515"})),(0,r.kt)("h2",{id:"unit-testing"},(0,r.kt)("a",{parentName:"h2",href:"https://github.com/jackmleitch/StravaDataPipline/tree/master/tests"},"Unit Testing")),(0,r.kt)("p",null,"Unit testing was performed using PyTest and all tests can be found in the tests directory. For example, below we see a unit test to test the ",(0,r.kt)("inlineCode",{parentName:"p"},"make_strava_api_request")," function. It asserts that a dictionary response is received and also that the response contains an 'id' key that is an integer."),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},'@pytest.mark.filterwarnings("ignore::urllib3.exceptions.InsecureRequestWarning")\ndef test_make_strava_api_request():\n    header = connect_strava()\n    response_json = make_strava_api_request(header=header, activity_num=1)\n    assert "id" in response_json.keys(), "Response dictionary does not contain id key."\n    assert isinstance(response_json, dict), "API should respond with a dictionary."\n    assert isinstance(response_json["id"], int), "Activity ID should be an integer."\n')))}h.isMDXComponent=!0},1229:function(t,e,a){e.Z=a.p+"assets/images/DAG-2835b36ad7768a84b9681567353d93b7.png"},2682:function(t,e,a){e.Z=a.p+"assets/images/dashboard-dc4fd7387c552922dee1c121e3deca45.png"},1336:function(t,e,a){e.Z=a.p+"assets/images/dashboard_map-f815f837c8278a1d3af8c575f875e40f.png"},6975:function(t,e,a){e.Z=a.p+"assets/images/slack_output-2cbea15fd35db870f4c152f271fa8314.png"},6527:function(t,e,a){e.Z=a.p+"assets/images/system_diagram-11c14c74c5c8bf3fa9760c95664ef611.png"}}]);