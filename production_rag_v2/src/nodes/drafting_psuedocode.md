DO NOT TOUCH NO MATTER WHAT

 """
        DETERMINISTIC LOGIC FLOW (CLINICALLY ROBUST VERSION):
        
        [STATE SCHEMA]
        notion_workspace = {
            "qa_db_id": UUID,        # Resolved Target
            "meeting_db_id": UUID,   # Resolved Target
            "page_id_qa": UUID,      # Container Coordinate
            "page_id_meeting": UUID, # Container Coordinate
            "qa_schema": dict,       # Mapping
            "meeting_schema": dict   # Mapping
        }
        COMMAND = user_input_since_from_the_receptionist

        # [PHASE 1: INTENTIONALLY DESIGNED ANALYSIS]
        # MCP-Agent analyzes COMMAND -> returns analysis_res
        analysis_res = {
            "is_exist_link" : bool,
            "links" : str | list[str] # the page link(s) that user's feed toward chat (first_link that match the default_notion_template link is the default link contain the DBs)
            "export_toward_what_db" : meeting | qa ,
            "selected_conv_ptr_to_post_toward_db" : list[str]
            "selected_conv_ptr_to_post_toward_meeting" : list[str]
        }


        is_Exist_link = check if exist notion_link from the COMMAND # COMMAND_SUGGEST_NEW_SAVING_LOCATION
        link = none # storing all the link & info

        
        if is_Exist_link:
            link = extract link from the COMMAND
            if match the notion domain
                -> is_exist_link == True
            else:
                -> is_exist_link = False
        
        # PHASE 1: DISCOVERY & INTENT
        # Check if notion_workspace exists in agent state.
        # IF empty OR user mentions "change", "move", "reset", "update":
            # IF Empty # the Notion_workspace
                ask user to provide an pre-existed notion-link for storing export-dirrection
                # Link can be either belong to :
                        a page inside the db 
                        a standalone db
                        link of a standalone page 
                        a pages in the DB-cell 
                link = interrupt(ask prompt to provide Notion Link of a pages / database)
                if link == empty: # user only press-enter
                    # create a new root page
                    # create a new DB with the following within the root
                else: # interrupt link is not empty
                    # INGESTION:
                        # categorizing the link: is it a page or a db  (Notion_API_Calling)
                            # -> setup case 1: if the link is a standalone DB - you have to call Notion API 
                                - to get the UUID of a parent-page contain the DB & extract the DB_UUID dirrectly from the link
                            # -> setup case 2: if the link is a page inside the db -> you have to call Notion API 
                                - to get the UUID of the db contain the page & get the UUID of the page from the link
                            # -> setup case 3: if the link is a standalone page (no DB dependencies connected) 
                                - take the page_id from the link & set db_uuid = Null
                            # => setup case 4: if the link is a DB-but come with a view 
                                - consider as the standalone DB case (the view is just a filter of the DB)
                        # check if the found DB's schema is sufficient to be qa_db
                            # if not -> add the missing attribute to the DB columns
                        # default set the page_id_qa == page_id_meeting
                        # check in the pages (default pick the page_id_qa) have 2 DB
                            # if not (only 1 DB)
                                # create another_db and set the new db -> meeting_db_id 
                            # else:
                                # pick the other_db -> meeting_db_id (the other db is the meeting db)

        # IF user mentions "change", "move", "reset", "update":
            
            ELSE: # if not note_workspace in agent's state not empty
                # check user's intention
                # check user's want to change the location of saving & export the Q&A / Meeting Note
                    if user want to change place database # if the intention want to change the database_id
                        -> INGESTION:
                            -> find the page_parent_id of the database_id
                            -> find the uuid of the database from the link
                            -> update database_id == uuid
                            -> update page_id_qa == page_parent_id
                    if user want to change place of meeting_db_id
                        -> INGESTION:
                            -> find the page_parent_id of the database_id
                            -> find the uuid of the database from the link
                            -> update database_id == uuid
                            -> update page_id_meeting == page_parent_id
                    
                
        #  Proceed to PHASE 2 (Dispatch/Publishing).
        # Determining 

        # FINAL VERIFICATION: Return immutable summary of transaction.
        """

        # Purpose: Workflow-only logic (LLM parses Natural Language into Deterministic JSON)
        
        analysis_res = analyze_command_with_llm(COMMAND) -> {
            "is_exist_link": bool,
            "links": list[str],
            "is_unclear_goal": bool,            # SET TRUE if NL command is ambiguous
            "export_targets": list[str],        # e.g., ["meeting", "qa"]
            "qa_data_pointers" : list[str],     # The specific Q&A messages to sync
            "meeting_data_pointers" : list[str] # The specific meeting summary messages to sync
        }

        # 1.1 AMBIGUITY GUARDRAIL
        if analysis_res["is_unclear_goal"]:
            RETURN { 
                "status": "fail", 
                "reason": "unclear_goal", 
                "message": "Failure to process due to unclear-goal. Please specify if you want to sync Q&A or Meeting notes." 
            }

        # [PHASE 2: DISCOVERY & SETUP]
        # (Only proceeds if Phase 1 passes)
        if notion_workspace is EMPTY:
            # ... Rest of your original Ingestion Cases 1-4 logic here ...
            link = resolve_or_interrupt(analysis_res["links"])
            
            # Category Check: DB vs Page resolve logic
            # Sibling Pairing Logic (Check for 2 DBs on page)
            # Bootstrap if companion is missing

        # [PHASE 3: PRE-FLIGHT VALIDATION]
        # Guardrail: Ensure all export_targets have non-null IDs in notion_workspace state.

        # [PHASE 4: DISPATCH]
        # Atomic loop to log data to resolved targets.
        """
