def handle(client, data=None, secrets=None, function_call_info=None):
    """Handler Function for failure prediction
    Args:
        client : Cognite Client (not needed, it's available to it, when deployed)
        data : data needed by function
        secrets : Any secrets it needs
        function_call_info : any other information about function

    Returns:
        response : response or result from the function

    [requirements]
    pandas
    scikit-learn
    [/requirements]
    """

    #download model 
    file_obj =  client.files.retrieve(external_id="rfc_model_rop_well5832")
    feature_name_list = file_obj.metadata['feature_list'].split(";")
    
    print(feature_name_list)
    # load data
    print("input data {}".format(data))
    
    df = client.sequences.data.retrieve_dataframe(external_id=data["external_id"],start=data["start"],end=data["end"])
    
    # load the model into memory
    loaded_model = pickle.loads(client.files.download_bytes(id=file_obj.id))
    # make the response serializable
    predictions = loaded_model.predict(df[feature_name_list]).tolist()
    return predictions
