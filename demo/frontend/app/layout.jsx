import './globals.css'

export const metadata = {
  title: 'AI Spell Checker - 99% Accuracy',
  description: 'Machine Learning powered spell checker with 99% accuracy',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}