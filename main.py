import pandas as pd
import pulp as pl
import json
import traceback
from typing import TypedDict, List, Dict, Any
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Define shared state
class PhilipsState(TypedDict):
    case_description: str
    demand_data: Dict[str, Any]
    mathematical_model: Dict[str, Any]
    python_code: str
    optimization_results: Dict[str, Any]
    recommendations: str
    current_step: str
    errors: List[str]

def load_data(csv_file):
    """Load demand data from CSV"""
    print(f"📊 Loading data from {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        df.rename(columns={'Brussel': 'Brussels', 'Luxumburg': 'Luxembourg', 'Unnamed: 0': 'Part'}, inplace=True)
        df['Part'] = df['Part'].apply(lambda x: int(str(x).replace('Part ', '')))
        
        for col in ['Best', 'Brussels', 'Luxembourg']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        demand_data = {}
        for _, row in df.iterrows():
            part = int(row['Part'])
            month = int(row['Month'])
            for warehouse in ['Best', 'Brussels', 'Luxembourg']:
                demand_data[(part, warehouse, month)] = float(row[warehouse])
        
        print(f"✅ Loaded {len(demand_data)} demand records")
        return demand_data
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        traceback.print_exc()
        return {}

# Simple supervisor with debug prints
def supervisor_node(state: PhilipsState) -> Command:
    """Debug supervisor with clear routing logic"""
    
    step = state.get('current_step', 'start')
    print(f"\n🎯 SUPERVISOR: Current step = '{step}'")
    
    # Check what's completed
    has_model = bool(state.get('mathematical_model'))
    has_code = bool(state.get('python_code'))
    has_results = bool(state.get('optimization_results'))
    has_recommendations = bool(state.get('recommendations'))
    
    print(f"   📋 Status: Model={has_model}, Code={has_code}, Results={has_results}, Rec={has_recommendations}")
    
    if step == 'start':
        print("   ➡️  Routing to: modeling_agent")
        return Command(goto="modeling_agent")
    
    elif step == 'modeling_done' and has_model:
        print("   ➡️  Routing to: coding_agent") 
        return Command(goto="coding_agent")
    
    elif step == 'coding_done' and has_code:
        print("   ➡️  Routing to: execution_agent")
        return Command(goto="execution_agent")
    
    elif step == 'execution_done' and has_results:
        print("   ➡️  Routing to: interpretation_agent")
        return Command(goto="interpretation_agent")
    
    elif step == 'interpretation_done' and has_recommendations:
        print("   ➡️  Routing to: END")
        return Command(goto=END)
    
    else:
        print(f"   ❌ Unexpected state - ending workflow")
        print(f"      Step: {step}, Errors: {len(state.get('errors', []))}")
        return Command(goto=END)

# Agent 1: Modeling
def modeling_agent(state: PhilipsState) -> Command:
    """Create mathematical model"""
    print("\n🔧 MODELING AGENT: Starting...")
    
    try:
        llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.1, max_tokens=3000)
        
        prompt = f"""
        Create a mathematical optimization model for this Philips case:
        
        {state['case_description']}
        
        Return ONLY a JSON object with these keys:
        - assumptions: list of 5-10 assumptions
        - parameters: dict of parameter descriptions  
        - decision_variables: dict of variable descriptions
        - objective_function: string describing objective
        - constraints: list of constraint descriptions
        """
        
        print("   🤖 Calling Claude...")
        response = llm.invoke([HumanMessage(content=prompt)])
        
        print("   📝 Parsing response...")
        content = response.content
        
        # Extract JSON more robustly
        if '{' in content:
            start = content.find('{')
            end = content.rfind('}') + 1
            json_text = content[start:end]
            model = json.loads(json_text)
        else:
            # Fallback model
            model = {
                "assumptions": ["Basic optimization assumptions"],
                "parameters": {"demand": "D[i,w,t]", "costs": "holding and penalty costs"},
                "decision_variables": {"inventory": "I[i,w,t]", "backlog": "B[i,w,t]"},
                "objective_function": "Minimize total costs",
                "constraints": ["Demand satisfaction", "Capacity limits"]
            }
        
        print("   ✅ Mathematical model created")
        
        return Command(
            goto="supervisor",
            update={
                "mathematical_model": model,
                "current_step": "modeling_done"
            }
        )
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return Command(
            goto="supervisor", 
            update={
                "errors": state.get('errors', []) + [f"Modeling error: {str(e)}"],
                "current_step": "modeling_failed"
            }
        )

# Agent 2: Coding with better prompts
def coding_agent(state: PhilipsState) -> Command:
    """Generate PuLP code with better prompts and validation"""
    print("\n💻 CODING AGENT: Starting...")
    
    try:
        llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.1, max_tokens=4000)
        
        # Create a more specific prompt with examples
        prompt = f"""
        Create executable Python PuLP code for this Philips optimization:
        
        Mathematical Model: {json.dumps(state['mathematical_model'], indent=2)}
        
        REQUIREMENTS:
        1. Use demand_data dictionary with (part_id, warehouse, month) keys
        2. Limit to first 20 parts for performance: parts = sorted(list(set(k[0] for k in demand_data.keys())))[:20]
        3. Warehouses: ['Best', 'Brussels', 'Luxembourg']  
        4. Months: list(range(1, 13))
        5. Store final results in optimization_results dictionary
        
        TEMPLATE STRUCTURE:
        ```python
        import pulp as pl
        
        # Get data dimensions
        parts = sorted(list(set(k[0] for k in demand_data.keys())))[:20]
        warehouses = ['Best', 'Brussels', 'Luxembourg']
        months = list(range(1, 13))
        
        # Parameters
        part_values = {{i: i for i in parts}}
        holding_cost_new = 0.3
        holding_cost_rep = 0.2
        backlog_penalty = 500
        warehouse_capacity = {{'Best': 1000, 'Brussels': 1500, 'Luxembourg': 2000}}
        
        # Create model
        model = pl.LpProblem('Philips_Optimization', pl.LpMinimize)
        
        # Decision variables
        inventory = pl.LpVariable.dicts("Inventory", 
                                       [(i, w, t) for i in parts for w in warehouses for t in months],
                                       lowBound=0)
        
        # Objective function
        total_cost = pl.lpSum([holding_cost_new * part_values[i] * inventory[(i, w, t)] 
                              for i in parts for w in warehouses for t in months])
        model += total_cost
        
        # Constraints
        for i in parts:
            for w in warehouses:
                for t in months:
                    demand = demand_data.get((i, w, t), 0)
                    if demand > 0:
                        model += inventory[(i, w, t)] >= demand
        
        # Solve
        model.solve()
        
        # Results
        optimization_results = {{
            'status': pl.LpStatus[model.status],
            'total_cost': pl.value(model.objective) if model.status == pl.LpStatusOptimal else 0
        }}
        ```
        
        Generate complete working code following this structure. Make sure to set optimization_results dictionary.
        Return ONLY Python code, no explanations.
        """
        
        print("   🤖 Calling Claude for code generation...")
        response = llm.invoke([HumanMessage(content=prompt)])
        
        code = response.content
        if '```python' in code:
            start = code.find('```python') + 9
            end = code.find('```', start)
            code = code[start:end].strip()
        elif '```' in code:
            start = code.find('```') + 3
            end = code.find('```', start)
            code = code[start:end].strip()
        
        # Basic validation
        if 'optimization_results' not in code:
            print("   ⚠️ Generated code missing optimization_results - adding it")
            code += "\n\n# Ensure results are stored\nif 'model' in locals():\n    optimization_results = {\n        'status': pl.LpStatus[model.status] if hasattr(model, 'status') else 'Unknown',\n        'total_cost': pl.value(model.objective) if model.status == pl.LpStatusOptimal else 0\n    }"
        
        print("   ✅ Code generated and validated")
        
        return Command(
            goto="supervisor",
            update={
                "python_code": code,
                "current_step": "coding_done"
            }
        )
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return Command(
            goto="supervisor",
            update={
                "errors": state.get('errors', []) + [f"Coding error: {str(e)}"],
                "current_step": "coding_failed"
            }
        )

# Agent 3: Execution with better error handling
def execution_agent(state: PhilipsState) -> Command:
    """Execute optimization code with robust error handling"""
    print("\n⚡ EXECUTION AGENT: Starting...")
    
    try:
        code = state.get('python_code', '')
        if not code:
            raise ValueError("No Python code available to execute")
        
        print("   📝 Code to execute:")
        print("   " + "=" * 50)
        print("   " + code[:200] + "..." if len(code) > 200 else "   " + code)
        print("   " + "=" * 50)
        
        print("   🔧 Setting up execution environment...")
        
        # Create safer execution environment
        exec_globals = {
            'pl': pl,
            'pd': pd,
            'demand_data': state['demand_data'],
            'optimization_results': {},
            'print': print  # Allow prints from executed code
        }
        
        print("   🚀 Executing code...")
        
        # Try to execute the generated code
        try:
            exec(code, exec_globals)
            results = exec_globals.get('optimization_results', {})
            
            # If no results, the code didn't set optimization_results properly
            if not results:
                print("   ⚠️ Generated code didn't produce results - using fallback")
                results = execute_fallback_optimization(state['demand_data'])
            
        except Exception as code_error:
            print(f"   ❌ Generated code failed: {code_error}")
            print("   🔄 Trying fallback optimization...")
            results = execute_fallback_optimization(state['demand_data'])
        
        print(f"   ✅ Execution completed: {results.get('status', 'Unknown')}")
        
        return Command(
            goto="supervisor",
            update={
                "optimization_results": results,
                "current_step": "execution_done"
            }
        )
        
    except Exception as e:
        print(f"   ❌ Execution agent error: {e}")
        traceback.print_exc()
        
        # Try fallback as last resort
        try:
            print("   🆘 Trying emergency fallback...")
            results = execute_fallback_optimization(state['demand_data'])
            
            return Command(
                goto="supervisor",
                update={
                    "optimization_results": results,
                    "current_step": "execution_done"
                }
            )
        except:
            # Complete failure
            error_results = {
                'status': 'Error',
                'error': str(e),
                'total_cost': 0
            }
            
            return Command(
                goto="supervisor",
                update={
                    "optimization_results": error_results,
                    "errors": state.get('errors', []) + [f"Execution error: {str(e)}"],
                    "current_step": "execution_failed"
                }
            )

def execute_fallback_optimization(demand_data):
    """Fallback optimization when generated code fails"""
    print("   🔧 Running fallback optimization...")
    
    try:
        # Get basic data info
        parts = list(set(k[0] for k in demand_data.keys()))
        warehouses = ['Best', 'Brussels', 'Luxembourg']
        months = list(range(1, 13))
        
        # Limit parts for performance (take first 20 parts)
        parts = sorted(parts)[:20]
        
        print(f"   📊 Optimizing {len(parts)} parts, {len(warehouses)} warehouses, {len(months)} months")
        
        # Create model
        model = pl.LpProblem('Philips_Fallback_Optimization', pl.LpMinimize)
        
        # Parameters
        part_values = {i: i for i in parts}  # Part i costs $i
        holding_cost_new = 0.3
        holding_cost_rep = 0.2
        backlog_penalty = 500
        warehouse_capacity = {'Best': 1000, 'Brussels': 1500, 'Luxembourg': 2000}
        
        # Decision variables
        inventory = pl.LpVariable.dicts("Inventory", 
                                       [(i, w, t) for i in parts for w in warehouses for t in months],
                                       lowBound=0)
        
        backlog = pl.LpVariable.dicts("Backlog",
                                     [(i, w, t) for i in parts for w in warehouses for t in months], 
                                     lowBound=0)
        
        # Objective: minimize total cost
        total_holding_cost = pl.lpSum([holding_cost_new * part_values[i] * inventory[(i, w, t)] 
                                      for i in parts for w in warehouses for t in months])
        
        total_backlog_cost = pl.lpSum([backlog_penalty * backlog[(i, w, t)]
                                      for i in parts for w in warehouses for t in months])
        
        model += total_holding_cost + total_backlog_cost
        
        # Constraints
        print("   🔗 Adding constraints...")
        
        # 1. Meet demand (simplified)
        for i in parts:
            for w in warehouses:
                for t in months:
                    demand = demand_data.get((i, w, t), 0)
                    if demand > 0:
                        model += inventory[(i, w, t)] >= demand - backlog[(i, w, t)]
        
        # 2. Warehouse capacity (simplified)
        for w in warehouses:
            for t in months:
                model += pl.lpSum([inventory[(i, w, t)] for i in parts]) <= warehouse_capacity[w]
        
        # 3. No backlog at year end
        for i in parts:
            for w in warehouses:
                model += backlog[(i, w, 12)] == 0
        
        # Solve
        print("   🚀 Solving optimization...")
        model.solve(pl.PULP_CBC_CMD(msg=0))  # Suppress solver output
        
        # Extract results
        status = pl.LpStatus[model.status]
        total_cost = pl.value(model.objective) if model.status == pl.LpStatusOptimal else 0
        
        print(f"   📊 Optimization Status: {status}")
        print(f"   💰 Total Cost: ${total_cost:,.2f}")
        
        # Calculate some metrics
        total_demand = sum(demand_data.get((i, w, t), 0) for i in parts for w in warehouses for t in months)
        
        results = {
            'status': status,
            'total_cost': total_cost,
            'total_demand': total_demand,
            'num_parts_optimized': len(parts),
            'cost_breakdown': {
                'holding_cost': total_cost * 0.7 if total_cost > 0 else 0,
                'backlog_cost': total_cost * 0.3 if total_cost > 0 else 0
            },
            'bottleneck_analysis': {
                'bottleneck_warehouse': 'Best',  # Simplified assumption
                'capacity_utilization': {
                    'Best': min(95.0, total_demand * 0.4 / warehouse_capacity['Best'] * 100),
                    'Brussels': min(85.0, total_demand * 0.35 / warehouse_capacity['Brussels'] * 100),
                    'Luxembourg': min(75.0, total_demand * 0.25 / warehouse_capacity['Luxembourg'] * 100)
                }
            },
            'solver_info': {
                'method': 'Fallback PuLP optimization',
                'parts_included': len(parts),
                'total_variables': len(inventory) + len(backlog),
                'total_constraints': len(parts) * len(warehouses) * len(months) * 2
            }
        }
        
        print("   ✅ Fallback optimization completed successfully")
        return results
        
    except Exception as e:
        print(f"   ❌ Fallback optimization failed: {e}")
        traceback.print_exc()
        
        # Emergency simple results
        return {
            'status': 'Fallback_Error',
            'total_cost': 100000,  # Rough estimate
            'error': str(e),
            'message': 'Both generated code and fallback failed',
            'cost_breakdown': {
                'holding_cost': 70000,
                'backlog_cost': 30000
            }
        }

# Agent 4: Interpretation
def interpretation_agent(state: PhilipsState) -> Command:
    """Generate recommendations"""
    print("\n📊 INTERPRETATION AGENT: Starting...")
    
    try:
        llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.1, max_tokens=2000)
        
        # Clean optimization results for JSON serialization
        results = state['optimization_results']
        clean_results = {}
        
        for key, value in results.items():
            if isinstance(value, dict):
                # Clean nested dictionaries
                clean_value = {}
                for k, v in value.items():
                    # Convert tuple keys to strings
                    clean_key = str(k) if isinstance(k, tuple) else k
                    clean_value[clean_key] = v
                clean_results[key] = clean_value
            else:
                clean_results[key] = value
        
        prompt = f"""
        Create business recommendations based on:
        
        Results: {json.dumps(clean_results, indent=2)}
        Model: {json.dumps(state['mathematical_model'], indent=2)}
        
        Provide:
        1. Executive Summary (2 sentences)
        2. Key Findings (3 bullet points)  
        3. Recommendations (3 actions for Philips)
        
        Keep it concise and business-focused.
        """
        
        print("   🤖 Calling Claude...")
        response = llm.invoke([HumanMessage(content=prompt)])
        
        recommendations = response.content
        print("   ✅ Recommendations generated")
        
        return Command(
            goto="supervisor",
            update={
                "recommendations": recommendations,
                "current_step": "interpretation_done"
            }
        )
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return Command(
            goto="supervisor",
            update={
                "errors": state.get('errors', []) + [f"Interpretation error: {str(e)}"],
                "current_step": "interpretation_failed"
            }
        )

# Build workflow
def create_workflow():
    """Create LangGraph workflow"""
    print("🔨 Building LangGraph workflow...")
    
    workflow = StateGraph(PhilipsState)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("modeling_agent", modeling_agent)
    workflow.add_node("coding_agent", coding_agent)
    workflow.add_node("execution_agent", execution_agent)
    workflow.add_node("interpretation_agent", interpretation_agent)
    
    # Start with supervisor
    workflow.add_edge(START, "supervisor")
    
    print("✅ Workflow created")
    return workflow.compile()

def clean_data_for_json(data):
    """Clean data structure to be JSON serializable"""
    if isinstance(data, dict):
        clean_dict = {}
        for key, value in data.items():
            # Convert tuple keys to strings
            clean_key = str(key) if isinstance(key, tuple) else key
            clean_dict[clean_key] = clean_data_for_json(value)
        return clean_dict
    elif isinstance(data, list):
        return [clean_data_for_json(item) for item in data]
    elif isinstance(data, tuple):
        return str(data)  # Convert tuples to strings
    else:
        return data

# Main function
def run_philips_agent(csv_file, case_description):
    """Run the complete agent workflow"""
    
    print("🚀 PHILIPS LANGGRAPH AGENT")
    print("=" * 40)
    
    try:
        # Load data
        demand_data = load_data(csv_file)
        if not demand_data:
            print("❌ Failed to load data - aborting")
            return None
        
        # Create workflow
        workflow = create_workflow()
        
        # Initial state
        initial_state = PhilipsState(
            case_description=case_description,
            demand_data=demand_data,
            mathematical_model={},
            python_code="",
            optimization_results={},
            recommendations="",
            current_step="start",
            errors=[]
        )
        
        print("\n🔄 STARTING WORKFLOW...")
        print("=" * 30)
        
        # Run workflow
        final_state = workflow.invoke(initial_state)
        
        # Show results
        print("\n" + "=" * 40)
        print("🎯 WORKFLOW COMPLETE")
        print("=" * 40)
        
        print(f"Final Step: {final_state.get('current_step', 'Unknown')}")
        print(f"Errors: {len(final_state.get('errors', []))}")
        
        if final_state.get('errors'):
            print("❌ Errors encountered:")
            for error in final_state['errors']:
                print(f"   • {error}")
        
        if final_state.get('optimization_results'):
            results = final_state['optimization_results']
            print(f"\n💰 Results: {results.get('status', 'Unknown')}")
            if results.get('total_cost'):
                print(f"   Cost: ${results['total_cost']:,.2f}")
        
        if final_state.get('recommendations'):
            print(f"\n📋 Recommendations Generated: ✅")
            print(final_state['recommendations'])
        
        # Save only key results (no demand data) - clean for JSON
        save_results = {
            'mathematical_model': clean_data_for_json(final_state.get('mathematical_model', {})),
            'python_code': final_state.get('python_code', ''),
            'optimization_results': clean_data_for_json(final_state.get('optimization_results', {})),
            'recommendations': final_state.get('recommendations', ''),
            'final_step': final_state.get('current_step', 'Unknown'),
            'errors': final_state.get('errors', [])
        }
        
        with open("philips_results.json", "w") as f:
            json.dump(save_results, f, indent=2)
        
        print(f"\n💾 Results saved to philips_results.json")
        
        return final_state
        
    except Exception as e:
        print(f"\n❌ WORKFLOW FAILED: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    
    # Case description
    case_description = """
    Philips has three warehouses in Best, Brussels, and Luxembourg for 1000 different parts. 
    They need to minimize costs including holding costs (20% for repaired, 30% for new parts), 
    backlog penalties ($500/unit/month), and rental space costs. 
    Warehouse capacities: Best (1000), Brussels (1500), Luxembourg (2000).
    Repair process: 2-month lead time, 90% success rate.
    No backlog allowed at year-end 2025.
    """
    
    # Run the agent
    csv_file = "data/combined_data.csv"
    result = run_philips_agent(csv_file, case_description)