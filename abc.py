 if comments:
            summary_desc = (
                "Handle the user's latest comment: '{comments}'.\n"
                "Review the technical state from the diagnosis and the history:\n{user_qa_pair_content}.\n"
                "Always return all error code and there message in reponse"
                "Provide a direct answer to the user's question.\n"
                "\n**CRITICAL**: Output MUST be valid JSON."
                "\n**CRITICAL**: Do not repeat technical diagnosis if the user is asking a simple follow-up."
                "\n**IMPORTANT**: The person raising this incident is not a direct customer, but a bank employee who works in one of the branches."
                "\n**IMPORTANT**: If an SR needs to be raised, clearly state 'An SR needs to be raised' — do NOT claim it has already been raised."
            )
        else:
            summary_desc = (
                "Review the technical logs from the diagnosis phase.\n"
                "Synthesize the findings into the required JSON format.\n"
                "\n**CRITICAL**: Always return all error code and there message in reponse"
                "\n**CRITICAL**: Mask all PII and remove backend technical jargon. User is not supposed to know about SQL errors."
                "\n**CRITICAL**: Verify if the issue is resolved w.r.t incident description: {incident_description}, mention yes/no."
                "\n**CRITICAL**: Do not ask repetitive questions to the user."
                "\n**IMPORTANT**: The person raising this incident is not a direct customer, but a bank employee who works in one of the branches."
                "\n**IMPORTANT**: If an SR needs to be raised, clearly state 'An SR needs to be raised' — do NOT claim it has already been raised."
            )

        summary = Task(
