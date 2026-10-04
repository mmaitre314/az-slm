// Linux Spot VM driven through Run Command (no inbound access). Evicted VMs are deleted, along with
// their OS disk, NIC and public IP. Two layers of auto-shutdown:
//  1. An in-VM idle watchdog (installed by cloud-init) that deletes or deallocates the VM through
//     its system-assigned managed identity after a few hours without activity.
//  2. Azure's built-in daily auto-shutdown schedule (deallocate) as a fallback.

@description('VM name; also used to name its NIC, public IP, OS disk and auto-shutdown schedule.')
param name string

param location string

@description('VM size. Must be Spot-capable in the region and fit the regional Spot vCPU quota.')
param vmSize string

@description('Spot needs an EA, pay-as-you-go or Sponsored subscription; Visual Studio (MSDN) subscriptions only allow Regular.')
@allowed([
  'Spot'
  'Regular'
])
param priority string = 'Spot'

param subnetId string

param adminUsername string = 'azureuser'

@description('SSH public key. Required by Azure for Linux VMs, but never used: VMs are driven via Run Command.')
param sshPublicKey string

@description('cloud-init (#cloud-config) document, as plain text.')
param cloudInit string

param osDiskSizeGB int = 64

@description('Daily fallback shutdown time (HHmm, UTC). The schedule deallocates the VM.')
param dailyShutdownTime string

@description('Unique per deployment (default: deployment time). Keeps role assignment names fresh when a VM name is reused, since the new VM gets a new identity.')
param deploymentStamp string = utcNow()

param tags object = {}

// Least privilege for the idle watchdog: read, deallocate and delete its own VM and the resources
// deleted with it. Built-in roles (Virtual Machine Contributor) would also allow Run Command as root
// and VM rewrites to any process on the VM that asks IMDS for a token.
resource selfShutdownRole 'Microsoft.Authorization/roleDefinitions@2022-04-01' = {
  name: guid(resourceGroup().id, 'azslm-vm-self-shutdown')
  properties: {
    roleName: 'azslm VM self-shutdown (${resourceGroup().name})'
    description: 'Lets a VM identity read, deallocate and delete its own VM and its OS disk, NIC and public IP.'
    type: 'customRole'
    permissions: [
      {
        actions: [
          'Microsoft.Compute/virtualMachines/read'
          'Microsoft.Compute/virtualMachines/deallocate/action'
          'Microsoft.Compute/virtualMachines/delete'
          'Microsoft.Compute/disks/read'
          'Microsoft.Compute/disks/delete'
          'Microsoft.Network/networkInterfaces/read'
          'Microsoft.Network/networkInterfaces/delete'
          'Microsoft.Network/publicIPAddresses/read'
          'Microsoft.Network/publicIPAddresses/delete'
        ]
        notActions: []
      }
    ]
    assignableScopes: [
      resourceGroup().id
    ]
  }
}

resource pip 'Microsoft.Network/publicIPAddresses@2024-05-01' = {
  name: 'pip-${name}'
  location: location
  tags: tags
  sku: {
    name: 'Standard'
  }
  properties: {
    publicIPAllocationMethod: 'Static'
    publicIPAddressVersion: 'IPv4'
  }
}

resource nic 'Microsoft.Network/networkInterfaces@2024-05-01' = {
  name: 'nic-${name}'
  location: location
  tags: tags
  properties: {
    enableAcceleratedNetworking: true
    ipConfigurations: [
      {
        name: 'ipconfig1'
        properties: {
          subnet: {
            id: subnetId
          }
          privateIPAllocationMethod: 'Dynamic'
          publicIPAddress: {
            id: pip.id
            properties: {
              deleteOption: 'Delete'
            }
          }
        }
      }
    ]
  }
}

resource vm 'Microsoft.Compute/virtualMachines@2024-07-01' = {
  name: name
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    priority: priority
    evictionPolicy: priority == 'Spot' ? 'Delete' : null
    billingProfile: priority == 'Spot'
      ? {
          maxPrice: -1 // never evict on price; pay the current Spot price up to the pay-as-you-go price
        }
      : null
    hardwareProfile: {
      vmSize: vmSize
    }
    storageProfile: {
      diskControllerType: 'NVMe'
      imageReference: {
        publisher: 'Canonical'
        offer: 'ubuntu-24_04-lts'
        sku: 'server'
        version: 'latest'
      }
      osDisk: {
        name: 'osdisk-${name}'
        createOption: 'FromImage'
        caching: 'ReadWrite'
        diskSizeGB: osDiskSizeGB
        deleteOption: 'Delete'
        managedDisk: {
          storageAccountType: 'Premium_LRS'
        }
      }
    }
    osProfile: {
      computerName: name
      adminUsername: adminUsername
      customData: base64(cloudInit)
      linuxConfiguration: {
        disablePasswordAuthentication: true
        provisionVMAgent: true
        ssh: {
          publicKeys: [
            {
              path: '/home/${adminUsername}/.ssh/authorized_keys'
              keyData: sshPublicKey
            }
          ]
        }
      }
    }
    networkProfile: {
      networkInterfaces: [
        {
          id: nic.id
          properties: {
            primary: true
            deleteOption: 'Delete'
          }
        }
      ]
    }
    securityProfile: {
      securityType: 'TrustedLaunch'
      uefiSettings: {
        secureBootEnabled: true
        vTpmEnabled: true
      }
    }
  }
}

resource osDisk 'Microsoft.Compute/disks@2024-03-02' existing = {
  name: 'osdisk-${name}'
}

// The idle watchdog deletes/deallocates its own VM: grant the custom role on the VM and on the OS disk,
// NIC and public IP, which are deleted with it (deleteOption: Delete).
resource vmRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  #disable-next-line use-stable-resource-identifiers
  name: guid(vm.id, 'azslm-vm-self-shutdown', deploymentStamp)
  scope: vm
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', selfShutdownRole.name)
    principalId: vm.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

resource diskRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  #disable-next-line use-stable-resource-identifiers
  name: guid(osDisk.id, vm.id, 'azslm-vm-self-shutdown', deploymentStamp)
  scope: osDisk
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', selfShutdownRole.name)
    principalId: vm.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

resource nicRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  #disable-next-line use-stable-resource-identifiers
  name: guid(nic.id, vm.id, 'azslm-vm-self-shutdown', deploymentStamp)
  scope: nic
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', selfShutdownRole.name)
    principalId: vm.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

resource pipRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  #disable-next-line use-stable-resource-identifiers
  name: guid(pip.id, vm.id, 'azslm-vm-self-shutdown', deploymentStamp)
  scope: pip
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', selfShutdownRole.name)
    principalId: vm.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

// Fallback: Azure's built-in daily auto-shutdown (deallocates the VM).
resource dailyShutdown 'Microsoft.DevTestLab/schedules@2018-09-15' = {
  name: 'shutdown-computevm-${name}'
  location: location
  tags: tags
  properties: {
    status: 'Enabled'
    taskType: 'ComputeVmShutdownTask'
    dailyRecurrence: {
      time: dailyShutdownTime
    }
    timeZoneId: 'UTC'
    targetResourceId: vm.id
    notificationSettings: {
      status: 'Disabled'
    }
  }
}

output vmName string = vm.name
