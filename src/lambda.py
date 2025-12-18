import json
import boto3
import uuid

def lambda_handler(event, context):
    #Create a client connection with Bedrock Agentcore
    client = boto3.client('bedrock-agentcore', region_name='us-west-2')

    #Get user input from event, match to the expected Agent payload structure
    user_input = event.get('prompt', 'Enter your question here...')
    payload = json.dumps({'prompt': user_input})

    #Generate a unique session ID
    session_id = f"lambda_session_{str(uuid.uuid4()).replace('-', '')}"
    print(payload, session_id)


    #invoke agent
    response = client.invoke_agent_runtime(
    agentRuntimeArn='arn:aws:bedrock-agentcore:us-west-2:181975986758:runtime/nasa_rag_4-v3BxAAGgMA',
    runtimeSessionId=session_id,  # Must be 33+ chars
    payload=payload,
    qualifier="DEFAULT" # Optional
    )

    #read and parse the response from AgentCore
    response_body = response['response'].read()
    response_data = json.loads(response_body)
    print("Agent Response:", response_data)

    #return successful Lambda response with CORS header
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',  # Adjust for specific origins if needed
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            'result': response_data,
            'session_id': session_id
        })
    }
