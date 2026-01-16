How to Use:

1) get_streams_holodex.py
    Get all the historical stream data from the Holodex API.
    This step usually takes a long time. Rate-limited by the API, so added a local delay that roughly matches it to avoid useless request spam.
    Suggest: Do the next manual steps while waiting for this to complete.
    Note: not tested recently

2) --- Manual: Make/edit denylist.json.
    List all the topics which should be ignored here. Example:
        Music_Cover
        Vlog
        asmr
        shorts
3) --- Manual: Make/edit ontology.txt.
    List all the tags which will be used to describe games. Example:
        fps
        competitive
        multiplayer

4) clean_json.py
    Clean the stream data and discard unnecessary info.
    Note: not tested recently

5) build_game_tags.py
    Creates a list (game_tags.json) of all the streamed games/topics, excluding those in the denylist.json.

6) --- Manual: Using ontology.txt created earlier, add tags to games in game_tags.json as appropriate.
        Ideally this should be done using a team of annotators, and then followed up with statistical measures such as Krippendorff's alpha to determine inter-rater reliability, etc., before proceeding.

7) build_graphs.py
    build_graphs.py - Build VTuber Collaboration and Game Interaction Graphs

Now we have a graph, so lots of analysis follows. For now, that's only one script. If this were to be used for a paper, statistical tests should be done to justify the prediction's usefulness.
8) predict.py
    Print a list of vtuber marketing suggestions

Extra:
inspect.py
    Get top stream weights overall or for specific vtuber. Useful for debug/verification/etc.


