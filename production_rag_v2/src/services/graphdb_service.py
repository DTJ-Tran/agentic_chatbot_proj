from neo4j import GraphDatabase
from typing import List, Dict, Any
from src.core.config import settings

class GraphDBService:
    """
    Handles graph-based storage for meeting entities and relationships.
    """
    def __init__(self):
        # Handle Neo4j URL normalization (bolt protocol usually runs on 7687)
        url = settings.neo4j_url
        if "7474" in url:
            # Replace HTTP port with Bolt port and protocol
            url = url.replace("7474", "7687").replace("http", "bolt")
        elif "localhost" in url and settings.server_domain != "localhost":
            # Fallback to server_domain if provided
            url = f"bolt://{settings.server_domain}:7687"
        
        self.driver = GraphDatabase.driver(
            url, 
            auth=(settings.neo4j_user, settings.neo4j_password)
        )

    def close(self):
        self.driver.close()

    def create_meeting_node(self, session_id: str, title: str, date: str):
        """Creates a Meeting node in Neo4j."""
        with self.driver.session() as session:
            session.execute_write(self._create_meeting, session_id, title, date)

    def link_meeting_participant_ephemeral(self, session_id: str, ephemeral_name: str):
        """Links an ephemeral speaker identity to a Meeting, avoiding global conflation."""
        with self.driver.session() as session:
            session.execute_write(self._link_participant_ephemeral, session_id, ephemeral_name)

    def link_concept(self, session_id: str, concept_name: str, concept_type: str):
        """Links a specialized Concept (technical term, internal code) to a Meeting."""
        with self.driver.session() as session:
            session.execute_write(self._link_concept, session_id, concept_name, concept_type)

    @staticmethod
    def _create_meeting(tx, session_id, title, date):
        query = (
            "MERGE (m:Meeting {sessionId: $session_id}) "
            "SET m.title = $title, m.date = $date "
            "RETURN m"
        )
        tx.run(query, session_id=session_id, title=title, date=date)

    @staticmethod
    def _link_participant_ephemeral(tx, session_id, name):
        # Creates a MeetingParticipant junction entity using a compound key to prevent edge-case collisions
        # Prepares resolving to a global SpeakerProfile optionally later.
        compound_id = f"{session_id}_{name}"
        query = (
            "MATCH (m:Meeting {sessionId: $session_id}) "
            "MERGE (mp:MeetingParticipant {id: $compound_id}) "
            "SET mp.ephemeral_name = $name "
            "MERGE (mp)-[:PARTICIPATED_IN]->(m)"
        )
        tx.run(query, session_id=session_id, compound_id=compound_id, name=name)

    @staticmethod
    def _link_concept(tx, session_id, name, type):
        query = (
            "MATCH (m:Meeting {sessionId: $session_id}) "
            "MERGE (c:Concept {name: $name}) "
            "SET c.conceptType = $type "
            "MERGE (m)-[:DISCUSSED]->(c)"
        )
        tx.run(query, session_id=session_id, name=name, type=type)

# Singleton instance
graphdb_service = GraphDBService()
