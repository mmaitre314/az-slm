// Shared per-region network: one VNet with one private subnet, guarded by an NSG with no inbound
// allow rules (the default rules deny inbound Internet traffic). VMs reach the Internet through
// their own Standard public IP; nothing needs inbound access because VMs are driven via Run Command.

@description('Azure region for the network.')
param location string

@description('Prefix for resource names.')
param namePrefix string = 'azslm'

param addressPrefix string = '10.42.0.0/16'
param subnetPrefix string = '10.42.0.0/24'
param tags object = {}

resource nsg 'Microsoft.Network/networkSecurityGroups@2024-05-01' = {
  name: 'nsg-${namePrefix}-${location}'
  location: location
  tags: tags
  properties: {
    securityRules: []
  }
}

resource vnet 'Microsoft.Network/virtualNetworks@2024-05-01' = {
  name: 'vnet-${namePrefix}-${location}'
  location: location
  tags: tags
  properties: {
    addressSpace: {
      addressPrefixes: [addressPrefix]
    }
    subnets: [
      {
        name: 'default'
        properties: {
          addressPrefix: subnetPrefix
          networkSecurityGroup: {
            id: nsg.id
          }
          defaultOutboundAccess: false
        }
      }
    ]
  }
}

output subnetId string = '${vnet.id}/subnets/default'
