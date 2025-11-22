<form method="post" action="/update_cases" style="margin:8px 0;">
  <label>How many days back?
    <select name="days">
      <option value="0">All filings</option>
      <option value="7">Last 7 days</option>
      <option value="14" selected>Last 14 days</option>
      <option value="30">Last 30 days</option>
      <option value="60">Last 60 days</option>
      <option value="90">Last 90 days</option>
      <option value="180">Last 180 days</option>
      <option value="365">Last 365 days</option>
    </select>
  </label>
  <button class="btn" type="submit">Update Cases</button>
</form>
