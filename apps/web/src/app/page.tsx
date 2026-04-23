import { redirect } from "next/navigation";

// Root route → go straight to the dashboard. No auth, no landing page.
export default function Home() {
  redirect("/dashboard");
}
