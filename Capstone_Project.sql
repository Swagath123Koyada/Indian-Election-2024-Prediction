Create database Capstone_Project

Use Capstone_Project

Select * from [dbo].[Party_Data]

Select * from [dbo].[Votes_Info]

                                                            SQL Project

1. Create a view to join both tables to display all results(columns).

      Create View FullElectionResults as
      Select p.ID,p.State,p.Constituency,p.Candidate,p.Party,p.Result,v.EVM_Votes,v.Postal_Votes,v.Total_Votes,v.Percentage_of_Votes
      from Party_Data p
      Join Votes_Info v ON p.ID = v.ID;

	  Select * from FullElectionResults;

2. Create a view, where the candidate won the election in their Constituency (value should be 'Yes' or 'No') along with all columns.

      Create View Election_Results as
      Select p.ID,p.State,p.Constituency,p.Candidate,p.Party,p.Result,v.EVM_Votes,v.Postal_Votes,v.Total_Votes,v.Percentage_of_Votes,
      Case
           When Result = 'Won' then 'Yes'
           Else 'No'
           End as Winning_Candidate
           from Party_Data p
      Join Votes_Info v ON p.ID = v.ID;

       Select * from Election_Results;


3. Find the candidate with the highest percentage of votes in each state.

      Select e1.State, e1.Candidate, e1.Party, e1.Percentage_of_Votes from FullElectionResults e1
      Join (
      Select State, Max(Percentage_of_Votes) as MaxPercentage from FullElectionResults
      Group by State
      ) e2 on e1.State = e2.State and e1.Percentage_of_Votes = e2.MaxPercentage
      Order by e1.State;


4. List all candidates who received more than the average total votes.

        Select CANDIDATE, Total_Votes from FullElectionResults
        Where Total_Votes > (Select Avg( Total_Votes ) from FullElectionResults);


5. Find the average EVM votes per state and then list all candidates who received higher than the average EVM votes in their state.

        Select Distinct(STATE), CANDIDATE, EVM_VOTES  from FullElectionResults A
        Where EVM_VOTES >
        (Select Avg(EVM_VOTES) as AVG_OF_EVM_VOTES from FullElectionResults B
        Where A.STATE = B.STATE)
        Order by STATE;


6. List pairs of candidates in the same constituency and the difference in their total votes.

        Select A.Candidate as Candidate1,B.Candidate as Candidate2, A.Total_Votes,
        Abs(A.Total_Votes - B.Total_Votes) as VoteDifference, A.Constituency from FullElectionResults A
        Join 
        FullElectionResults B on A.Constituency = B.Constituency and A.Candidate <> B.Candidate;


7. Find pairs of candidates in the same state who have similar percentages of votes (within 1% difference).

        Select A.Candidate as Candidate1,B.Candidate as Candidate2,A.State,
        Abs(A.Percentage_of_Votes - B.Percentage_of_Votes) as Percentage_Difference from FullElectionResults A
        Join 
        FullElectionResults B on A.State = B.State 
        Where A.Candidate <> b.Candidate and Abs(A.Percentage_of_Votes - B.Percentage_of_Votes) <= 1;


8. List pairs of candidates from the same party along with their constituencies and total votes.

        Select A.Candidate as Candidate1,B.Candidate as Candidate2, A.Constituency,A.Total_Votes as Votes1,B.Total_Votes as Votes2,A.Party from FullElectionResults A
        Join 
        FullElectionResults B on A.Party = B.Party 
        Where A.Candidate <> B.Candidate and A.Constituency = B.Constituency;


9. Find the candidates within the same party who have the maximum and minimum total votes in each state.
    
	    Select e1.State, e1.Party, e1.Candidate as MaxVotesCandidate, e1.Total_Votes as MaxTotalVotes from FullElectionResults e1
        Join (
        Select State, Party, Max(Total_Votes) as MaxTotalVotes from FullElectionResults
        Group by State, Party
        ) e2 on e1.State = e2.State and e1.Party = e2.Party and e1.Total_Votes = e2.MaxTotalVotes
        Order by e1.State, e1.Party;

        Select e1.State, e1.Party, e1.Candidate as MinVotesCandidate, e1.Total_Votes as MinTotalVotes from ElectionResults e1
        Join (
        Select State,Party, Min(Total_Votes) as MinTotalVotes from ElectionResults
        Group by State, Party
        ) e2 on e1.State = e2.State and e1.Party = e2.Party and e1.Total_Votes = e2.MinTotalVotes
        Order by e1.State, e1.Party;

10. Find the difference in ranks between the total votes and the percentage of votes for each candidate within their constituency.

        Select ID,State,Constituency,Candidate,Party,Total_Votes,Percentage_of_Votes,
        Rank() over (Partition by Constituency Order by Total_Votes desc) as Rank_Total_Votes,
        Rank() over (Partition by Constituency Order by Percentage_of_Votes desc) as Rank_Percentage_of_Votes,
        Rank() over (Partition by Constituency Order by Total_Votes desc) - 
        Rank() over (Partition by Constituency Order by Percentage_of_Votes desc) as Rank_Difference
        from FullElectionResults


11. Find the total votes of the previous candidate within each constituency based on the total votes.

        Select ID,State,Constituency,Candidate,Party,Total_Votes,
        Lag(Total_Votes) over (Partition by Constituency Order by Total_Votes desc) as Previous_Total_Votes
        from FullElectionResults;


12. Find the winning margin (difference in total votes) between the top two candidates in each constituency.

        With RankedCandidates as (
        Select p.ID,p.State,p.Constituency,p.Candidate,p.Party,v.Total_Votes,
        Rank() over (Partition by p.Constituency Order by v.Total_Votes desc) as Rank
        from Party_Data p
        Join Votes_Info v on p.ID = v.ID
      )
        Select 
        rc1.Constituency,
        rc1.Candidate as Winner,
        rc1.Total_Votes as Winner_Total_Votes,
        rc2.Candidate as Runner_Up,
        rc2.Total_Votes as Runner_Up_Total_Votes,
       (rc1.Total_Votes - rc2.Total_Votes) as Winning_Margin
        from RankedCandidates rc1
        Join RankedCandidates rc2 on rc1.Constituency = rc2.Constituency and rc2.Rank = 2
        where rc1.Rank = 1;

13. Calculate the percentage of total votes each candidate received out of the total votes in their state and list the candidates along with their calculated percentage.

        With StateTotalVotes as (
        Select p.State,SUM(v.Total_Votes) AS Total_State_Votes from Party_Data p
        Join Votes_info v on p.ID = v.ID
        Group by p.State
      )
        Select p.State,p.Constituency,p.Candidate,p.Party,v.Total_Votes,stv.Total_State_Votes,(v.Total_Votes * 100.0 / stv.Total_State_Votes) AS Percentage_of_State_Votes
        from Party_Data p
        Join Votes_info v on p.ID = v.ID
        Join StateTotalVotes stv on p.State = stv.State;



14. Calculate the share of total votes each candidate received out of the total votes in their state.

        Select State, Candidate, Total_Votes,Sum(Total_Votes) over (Partition by State) as State_Total_Votes,
       (Round((Total_Votes * 100.0 / (Select Sum (Total_Votes) from FullElectionResults B Where B.State = A.State)),2)) as Vote_Share from FullElectionResults A;


15. List all constituencies where the difference in total votes between the winner and the runner-up is less than 5%.


        Select e1.State, e1.Constituency, e1.Candidate as Winner, e1.Total_Votes as WinnerVotes, e2.Candidate as RunnerUp, e2.Total_Votes as RunnerUpVotes,
        ((e1.Total_Votes - e2.Total_Votes) * 100.0 / e1.Total_Votes) as VoteDifferencePercentage 
	    from (
        Select State, Constituency, Candidate, Total_Votes,
        Rank() over (Partition by State, Constituency Order by Total_Votes desc) as VoteRank
        from ElectionResults
       ) e1
        Join (
        Select State, Constituency, Candidate, Total_Votes,
        Rank() over (Partition by State, Constituency Order by Total_Votes desc) as VoteRank
        from ElectionResults
       ) e2 on e1.State = e2.State and e1.Constituency = e2.Constituency and e1.VoteRank = 1 and e2.VoteRank = 2
        Where ((e1.Total_Votes - e2.Total_Votes) * 100.0 / e1.Total_Votes) < 5
        Order by e1.State, e1.Constituency;
