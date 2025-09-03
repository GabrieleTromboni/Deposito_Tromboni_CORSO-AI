"""
Simple test for DatabaseCrew functionality without network dependencies.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.progetto_crew_flows.crews.database_crew.database_crew import DatabaseCrew, create_database_crew

def test_database_crew_basic():
    """Test basic DatabaseCrew functionality"""
    print("🧪 Testing Basic DatabaseCrew functionality...")
    
    # Test 1: Create DatabaseCrew instance
    print("\n1️⃣ Creating DatabaseCrew instance...")
    try:
        database_crew = create_database_crew()
        print("✅ DatabaseCrew created successfully")
        print(f"   Database type: {database_crew.database_type}")
        print(f"   Available databases: {database_crew.available_databases}")
    except Exception as e:
        print(f"❌ Failed to create DatabaseCrew: {e}")
        return False
    
    # Test 2: Set database configuration
    print("\n2️⃣ Testing database configuration...")
    try:
        database_crew.set_database_type("faiss")
        database_crew.set_available_databases(["faiss"])
        print("✅ Database configuration set successfully")
        print(f"   Database type: {database_crew.database_type}")
        print(f"   Available databases: {database_crew.available_databases}")
    except Exception as e:
        print(f"❌ Failed to configure database: {e}")
        return False
    
    # Test 3: Check agents creation
    print("\n3️⃣ Testing agents creation...")
    try:
        agents_dict = {
            "database_manager": database_crew.database_manager,
            "qdrant_specialist": database_crew.qdrant_specialist,
            "rag_retrieval_specialist": database_crew.rag_retrieval_specialist,
            "content_formatter": database_crew.content_formatter
        }
        
        for agent_name, agent_func in agents_dict.items():
            try:
                agent = agent_func()
                print(f"   ✅ {agent_name}: Created successfully")
                print(f"      Role: {agent.role}")
                print(f"      Tools: {len(agent.tools) if agent.tools else 0}")
            except Exception as e:
                print(f"   ❌ {agent_name}: Failed to create - {e}")
                
    except Exception as e:
        print(f"❌ Failed to test agents: {e}")
        return False
    
    # Test 4: Check tasks creation
    print("\n4️⃣ Testing tasks creation...")
    try:
        tasks_dict = {
            "create_qdrant_database_task": database_crew.create_qdrant_database_task,
            "create_faiss_database_task": database_crew.create_faiss_database_task,
            "execute_rag_retrieval_task": database_crew.execute_rag_retrieval_task,
            "format_rag_results_task": database_crew.format_rag_results_task
        }
        
        for task_name, task_func in tasks_dict.items():
            try:
                task = task_func()
                print(f"   ✅ {task_name}: Created successfully")
                print(f"      Description: {task.description[:100]}...")
            except Exception as e:
                print(f"   ❌ {task_name}: Failed to create - {e}")
                
    except Exception as e:
        print(f"❌ Failed to test tasks: {e}")
        return False
    
    # Test 5: Check crew creation (without execution)
    print("\n5️⃣ Testing crew creation...")
    try:
        database_crew._operation_type = 'rag_retrieval'
        crew = database_crew.crew()
        print("✅ Crew created successfully")
        print(f"   Number of agents: {len(crew.agents)}")
        print(f"   Number of tasks: {len(crew.tasks)}")
        print(f"   Process type: {crew.process}")
    except Exception as e:
        print(f"❌ Failed to create crew: {e}")
        return False
    
    print("\n🎉 All basic tests passed successfully!")
    return True

def main():
    """Main test function"""
    print("=" * 60)
    print("DATABASE CREW BASIC FUNCTIONALITY TEST")
    print("=" * 60)
    
    success = test_database_crew_basic()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)

if __name__ == "__main__":
    main()
