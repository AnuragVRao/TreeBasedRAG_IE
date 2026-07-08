export function Header() {
  return (
    <header className="border-b border-border bg-background/80 backdrop-blur-sm">
      <div className="mx-auto px-8 py-4">
        <div className="flex items-center gap-3">
          <img src="/logo.png" alt="EywaAI logo" className="h-8 w-8 rounded-lg object-cover" />
          <div>
            <h1 className="text-lg font-medium tracking-tight text-foreground">
              EywaAI
            </h1>
          </div>
        </div>
      </div>
    </header>
  );
}